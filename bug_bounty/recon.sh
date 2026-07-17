#!/bin/bash
# Bug Bounty Recon Script — Beginner Target
# Target: A small-to-medium open source project with bounty program

TARGET="github.com"
OUTDIR="recon_$TARGET"
mkdir -p "$OUTDIR"

echo "=== Recon for $TARGET ==="
echo ""

# Step 1: Check if tools are available
echo "[1/4] Checking tools..."
which subfinder 2>/dev/null && echo "  subfinder: OK" || echo "  subfinder: MISSING (install: go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest)"
which httpx 2>/dev/null && echo "  httpx: OK" || echo "  httpx: MISSING (install: go install -v github.com/projectdiscovery/httpx/cmd/httpx@latest)"
which nuclei 2>/dev/null && echo "  nuclei: OK" || echo "  nuclei: MISSING (install: go install -v github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest)"
which ffuf 2>/dev/null && echo "  ffuf: OK" || echo "  ffuf: MISSING (install: apt install ffuf)"
which curl 2>/dev/null && echo "  curl: OK" || echo "  curl: MISSING"
which nmap 2>/dev/null && echo "  nmap: OK" || echo "  nmap: MISSING"

echo ""
echo "[2/4] Starting basic recon..."

# Step 2: Simple HTTP probe
echo "  Probing target..."
curl -s -o /dev/null -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\nRedirect: %{redirect_url}\n" "https://$TARGET" > "$OUTDIR/http_probe.txt" 2>&1
cat "$OUTDIR/http_probe.txt"

# Step 3: Check common endpoints
echo ""
echo "[3/4] Checking common endpoints..."
for path in /robots.txt /sitemap.xml /.well-known/security.txt /api /admin /debug /health /status; do
    status=$(curl -s -o /dev/null -w "%{http_code}" "https://$TARGET$path" 2>/dev/null)
    echo "  $path → $status"
    echo "$path → $status" >> "$OUTDIR/endpoints.txt"
done

# Step 4: Check for common misconfigurations
echo ""
echo "[4/4] Checking headers..."
curl -sI "https://$TARGET" > "$OUTDIR/headers.txt" 2>&1
echo "  Security headers:"
grep -i "x-frame-options\|x-content-type\|strict-transport\|content-security\|x-xss-protection" "$OUTDIR/headers.txt" 2>/dev/null || echo "  (no security headers found)"

echo ""
echo "=== Recon Complete ==="
echo "Results saved to: $OUTDIR/"
