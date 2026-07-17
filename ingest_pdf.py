"""ingest_pdf.py — Ingest any document into AGI knowledge base.

Supports:
  python3 ingest_pdf.py /path/to/file.pdf              # local file
  python3 ingest_pdf.py --url <url>                      # download from URL
  python3 ingest_pdf.py --arxiv 2302.13971               # arXiv paper
  python3 ingest_pdf.py --search "quantum field theory"  # search + download
  python3 ingest_pdf.py --list                           # list ingested books
"""

import argparse, json, math, os, re, sys, time, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

BASE = Path(__file__).parent
_fitz = None
_embed = None

def get_embedding(text: str) -> list[float]:
    global _embed
    if _embed is None:
        import axiom
        _embed = axiom._embed
    return _embed(text[:2048])

def _lazy_fitz():
    global _fitz
    if _fitz is None:
        import fitz
        _fitz = fitz
    return _fitz

def extract_text(path: str) -> str:
    fitz = _lazy_fitz()
    doc = fitz.open(path)
    total = len(doc)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(text)
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{total} pages ({len("".join(pages))//1000}KB)', flush=True)
    return '\n\n'.join(pages)

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[tuple[str, str]]:
    paragraphs = text.split('\n\n')
    chunks = []
    current = ''
    start_title = ''
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > chunk_size and current:
            chunks.append((start_title[:80], current.strip()))
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + '\n\n' + para
        else:
            if not current:
                start_title = para[:120]
            current += '\n\n' + para
    if current.strip():
        chunks.append((start_title[:80], current.strip()))
    return chunks

def download_url(url: str, dest: Path) -> str:
    print(f'  downloading {url}...')
    req = urllib.request.Request(url, headers={'User-Agent': 'AxiomAGI/1.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    dest.write_bytes(data)
    print(f'  saved {len(data)//1024}KB to {dest.name}')
    return str(dest)

def search_pdf(topic: str) -> list[dict]:
    """Search for free PDFs on a topic via Google and direct sources."""
    results = []
    # Try arXiv search
    try:
        query = quote(f'all:{topic}')
        url = f'http://export.arxiv.org/api/query?search_query={query}&max_results=5&sortBy=relevance'
        req = urllib.request.Request(url, headers={'User-Agent': 'AxiomAGI/1.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            xml_data = r.read().decode()
        root = ET.fromstring(xml_data)
        ns = {'a': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('a:entry', ns):
            title = entry.find('a:title', ns).text.strip()[:200] if entry.find('a:title', ns) is not None else topic
            arxiv_id = entry.find('a:id', ns).text.split('/')[-1] if entry.find('a:id', ns) is not None else ''
            pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf' if arxiv_id else ''
            results.append({'title': title, 'url': pdf_url, 'source': 'arxiv'})
    except Exception as e:
        print(f'  arxiv search failed: {e}')
    # Try direct PDF search
    try:
        search_q = quote(f'{topic} filetype:pdf')
        gurl = f'https://www.google.com/search?q={search_q}'
        req = urllib.request.Request(gurl, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode(errors='replace')
        for m in re.finditer(r'href="(https?://[^"]+\.pdf)"', html):
            url = m.group(1)
            if any(ext in url for ext in ['.pdf']):
                results.append({'title': url.split('/')[-1][:200], 'url': url, 'source': 'web'})
    except Exception:
        pass
    return results

def store_in_db(chunks: list[tuple[str, str]], source: str, db: Path):
    import sqlite3
    con = sqlite3.connect(str(db))
    # Ensure column exists
    cols = [r[1] for r in con.execute('PRAGMA table_info(chunks)').fetchall()]
    if 'created_at' not in cols:
        con.execute('ALTER TABLE chunks ADD COLUMN created_at REAL')
    con.execute('''CREATE TABLE IF NOT EXISTS chunks
        (embedding BLOB, source TEXT, url TEXT, title TEXT, content TEXT, created_at REAL)''')
    text_col = 'content'
    t0 = time.time()
    stored = 0
    for i, (title, chunk_text) in enumerate(chunks):
        try:
            vec = get_embedding(chunk_text)
            con.execute(f'INSERT INTO chunks (embedding, source, url, title, {text_col}, created_at) VALUES (?,?,?,?,?,?)',
                        (json.dumps(vec), source, source, title, chunk_text[:2000], time.time()))
            stored += 1
        except Exception as e:
            print(f'  chunk {i} error: {e}')
        if (i + 1) % 50 == 0:
            con.commit()
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(chunks) - i - 1) / max(rate, 0.01)
            print(f'  {i+1}/{len(chunks)} embedded ({rate:.0f}/s, ETA {eta:.0f}s)', flush=True)
    con.commit()
    con.close()
    print(f'  stored {stored}/{len(chunks)} chunks in {time.time()-t0:.0f}s')

def list_ingested(db: Path):
    import sqlite3
    if not db.exists():
        print('  no knowledge base')
        return
    con = sqlite3.connect(str(db))
    # Check if created_at column exists
    cols = [r[1] for r in con.execute('PRAGMA table_info(chunks)').fetchall()]
    if 'created_at' in cols:
        rows = con.execute('SELECT source, COUNT(*), MAX(created_at) FROM chunks GROUP BY source ORDER BY source').fetchall()
        fmt = lambda ts: time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "?"
    else:
        rows = con.execute('SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source').fetchall()
        fmt = lambda ts: "?"
    con.close()
    print(f'  {"source":<25} {"chunks":<8} last_ingested')
    print(f'  {"-"*50}')
    for row in rows:
        s, cnt = row[0], row[1]
        ts = row[2] if len(row) > 2 else None
        print(f'  {s:<25} {cnt:<8} {fmt(ts)}')

def main():
    parser = argparse.ArgumentParser(description='Ingest a document into AGI knowledge base')
    parser.add_argument('path', nargs='?', help='Local file path')
    parser.add_argument('--url', help='Download PDF from URL')
    parser.add_argument('--arxiv', help='arXiv paper ID (e.g. 2302.13971)')
    parser.add_argument('--search', help='Search for PDFs on a topic')
    parser.add_argument('--source', default='', help='Source tag (default: filename or topic)')
    parser.add_argument('--chunk', type=int, default=2000, help='Chunk size in chars')
    parser.add_argument('--list', action='store_true', help='List ingested sources')
    args = parser.parse_args()

    db = BASE / '.axiom_state' / 'knowledge.db'

    if args.list:
        list_ingested(db)
        return

    # Determine source and file path
    source = args.source
    file_path = None
    tmp_dir = BASE / '.axiom_state' / 'ingested'
    tmp_dir.mkdir(exist_ok=True)

    if args.url:
        fname = args.url.split('/')[-1].split('?')[0] or 'download.pdf'
        dest = tmp_dir / fname
        download_url(args.url, dest)
        file_path = str(dest)
        source = source or fname.replace('.pdf', '')[:80]
    elif args.arxiv:
        url = f'https://arxiv.org/pdf/{args.arxiv}.pdf'
        dest = tmp_dir / f'{args.arxiv}.pdf'
        download_url(url, dest)
        file_path = str(dest)
        source = source or f'arxiv_{args.arxiv}'
    elif args.search:
        print(f'Searching for: {args.search}')
        results = search_pdf(args.search)
        if not results:
            print('  no PDFs found')
            return
        print(f'  found {len(results)} results')
        for i, r in enumerate(results[:5]):
            print(f'  [{i+1}] {r["title"][:80]}')
            print(f'       {r["source"]}: {r["url"][:80]}')
        # Auto-pick first result
        pick = results[0]
        print(f'\n  downloading: {pick["title"][:80]}')
        dest = tmp_dir / f'{pick["source"]}_{i}.pdf'
        try:
            download_url(pick['url'], dest)
            file_path = str(dest)
            source = source or pick['title'][:80]
        except Exception as e:
            print(f'  download failed: {e}')
            return
    elif args.path:
        file_path = args.path
        source = source or Path(args.path).stem[:80]
    else:
        parser.print_help()
        return

    if not file_path or not os.path.exists(file_path):
        print(f'  file not found: {file_path}')
        return

    print(f'Extracting: {file_path}')
    t0 = time.time()
    text = extract_text(file_path)
    print(f'  {len(text)//1000}KB text in {time.time()-t0:.1f}s')

    chunks = chunk_text(text, chunk_size=args.chunk)
    print(f'  {len(chunks)} chunks (avg {sum(len(c[1]) for c in chunks)//max(1,len(chunks))} chars)')

    store_in_db(chunks, source, db)
    print(f'  source="{source}" ingested')

if __name__ == '__main__':
    main()
