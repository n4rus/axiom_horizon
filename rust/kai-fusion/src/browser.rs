//! Headless browser via HTTP using ureq (lightweight, no GUI needed).
//!
//! Provides `kai browse <url>` to fetch pages and extract text content,
//! and `kai search <query>` to search DuckDuckGo.
//!
//! Designed for resource-constrained environments (5GB VRAM, 10GB RAM, CPU-bounded).

use std::time::Duration;

const USER_AGENT: &str = "Kai-Fusion/0.1";
const HTTP_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_BODY_SIZE: usize = 1_048_576; // 1 MB

/// Fetch a URL and extract readable text content (strips HTML tags).
pub fn browse_url(url: &str) -> Result<String, String> {
    let response = ureq::get(url)
        .set("User-Agent", USER_AGENT)
        .timeout(HTTP_TIMEOUT)
        .call()
        .map_err(|e| format!("HTTP error fetching {url}: {e}"))?;

    let content_type = response
        .header("Content-Type")
        .unwrap_or("text/html")
        .to_string();

    // ureq Response into_string() consumes the response and returns String
    let body = response
        .into_string()
        .map_err(|e| format!("Failed to read body from {url}: {e}"))?;

    if body.len() > MAX_BODY_SIZE {
        return Err(format!(
            "Response too large ({} bytes, max {}), try a more specific URL",
            body.len(),
            MAX_BODY_SIZE
        ));
    }

    let text = if content_type.contains("html") {
        html_to_text(&body)
    } else if content_type.contains("text") || content_type.is_empty() {
        body.clone()
    } else {
        format!("[binary content type: {content_type}, {:.1} KB]", body.len() as f32 / 1024.0)
    };

    Ok(text)
}

/// Search DuckDuckGo for a query and return top results as plain text.
/// Uses DuckDuckGo Lite (HTML-based, no API key needed).
pub fn search(query: &str) -> Result<String, String> {
    // Simple URL encoding: replace spaces with + and encode problematic chars
    let encoded = query
        .replace(' ', "+")
        .replace("&", "%26")
        .replace("=", "%3D")
        .replace("?", "%3F");
    let url = format!("https://lite.duckduckgo.com/lite/?q={encoded}");

    let response = ureq::get(&url)
        .set("User-Agent", USER_AGENT)
        .timeout(HTTP_TIMEOUT)
        .call()
        .map_err(|e| format!("search HTTP error: {e}"))?;

    let body = response
        .into_string()
        .map_err(|e| format!("search read error: {e}"))?;

    let results = extract_ddg_results(&body);
    if results.is_empty() {
        Ok(format!("No results found for '{}'", query))
    } else {
        let mut output = format!("Search '{}' - {} result(s):\n\n", query, results.len());
        for (i, (title, url_part, snippet)) in results.into_iter().enumerate() {
            let n = i + 1;
            output.push_str(&format!("{n}. {title}\n   {url_part}\n   {snippet}\n\n"));
        }
        Ok(output)
    }
}

/// Very simple HTML tag stripper: removes <script> and <style> blocks,
/// then replaces remaining tags with text, normalizing whitespace.
fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len().min(50_000));
    let mut chars = html.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '<' {
            // Read up to '>' to get the tag name
            let mut tag = String::new();
            let mut is_closing = false;
            let mut done = false;
            while let Some(&next) = chars.peek() {
                chars.next(); // consume
                if next == '>' {
                    done = true;
                    break;
                }
                if next == '/' && tag.is_empty() {
                    is_closing = true;
                } else if next != ' ' && next != '\t' && next != '\n' && next != '\r' {
                    tag.push(next);
                }
            }
            let tag_lower = tag.to_lowercase();
            // Skip content inside script and style tags
            if !is_closing && (tag_lower == "script" || tag_lower == "style" || tag_lower.starts_with("script") || tag_lower.starts_with("style")) {
                // Skip until </script> or </style>
                let skip_tag = format!("</{}", tag_lower);
                let mut skip_buf = String::new();
                while let Some(&next) = chars.peek() {
                    let ch = chars.next().unwrap();
                    skip_buf.push(ch);
                    if skip_buf.len() > skip_tag.len() + 10 {
                        skip_buf.remove(0);
                    }
                    if skip_buf.contains(&skip_tag) {
                        break;
                    }
                }
            }
        } else if c == '&' {
            // Handle HTML entities
            let mut entity = String::new();
            let mut semicolon_found = false;
            for _ in 0..10 {
                if let Some(&next) = chars.peek() {
                    chars.next();
                    if next == ';' {
                        semicolon_found = true;
                        break;
                    }
                    entity.push(next);
                } else {
                    break;
                }
            }
            if semicolon_found {
                let decoded = match entity.as_str() {
                    "amp" => "&",
                    "lt" => "<",
                    "gt" => ">",
                    "quot" => "\"",
                    "apos" => "'",
                    "nbsp" => " ",
                    _ => &format!("&{entity};"),
                };
                out.push_str(decoded);
            } else {
                out.push('&');
            }
        } else if c.is_ascii_control() && c != '\n' && c != '\t' {
            // Skip control chars but convert newline/tab normally
        } else {
            out.push(c);
        }
    }

    // Normalize whitespace
    normalize_whitespace(&out)
}

fn normalize_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_space = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !prev_space {
                if !result.is_empty() {
                    result.push('\n');
                }
                prev_space = true;
            }
        } else {
            if prev_space && !result.is_empty() {
                result.push('\n');
            }
            // Collapse internal whitespace in the line
            let mut words = trimmed.split_whitespace();
            if let Some(first) = words.next() {
                result.push_str(first);
                for word in words {
                    result.push(' ');
                    result.push_str(word);
                }
            }
            prev_space = false;
        }
    }

    result
}

/// Extract DuckDuckGo Lite search results from HTML.
fn extract_ddg_results(html: &str) -> Vec<(String, String, String)> {
    let mut results = Vec::new();

    // DuckDuckGo Lite: results are in <a class="result__a" href="URL">Title</a>
    // followed by <a class="result__snippet">Snippet</a>
    let tag_start = "result__a\"";
    let snippet_tag = "result__snippet\"";

    let mut pos = 0;
    while pos + tag_start.len() < html.len() {
        if let Some(found) = html[pos..].find(tag_start) {
            let abs = pos + found;
            let after = &html[abs..];

            // Extract href
            let href_start = after.find("href=\"").unwrap_or(0);
            let after_href = &after[href_start + 6..];
            let href_end = after_href.find('"').unwrap_or(after_href.len());
            let href = &after_href[..href_end];

            // Extract title (between > and </a>)
            let after_tag = &after_href[href_end + 1..];
            let title_start = after_tag.find('>').unwrap_or(0);
            let title_content = &after_tag[title_start + 1..];
            let title_end = title_content.find("</a>").unwrap_or(title_content.len());
            let title = title_content[..title_end].trim().to_string();

            // Get snippet from nearby area
            let search_area = if abs + found + 5000 < html.len() {
                &html[abs + found + 5000..(abs + found + 8000).min(html.len())]
            } else {
                ""
            };
            let snippet = if let Some(si) = search_area.find(snippet_tag) {
                let sa = &search_area[si + snippet_tag.len()..];
                if let Some(gt) = sa.find('>') {
                    let sc = &sa[gt + 1..];
                    if let Some(ca) = sc.find("</a>") {
                        sc[..ca].trim().to_string()
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            results.push((title, href.to_string(), snippet));
            pos = abs + found + tag_start.len();
        } else {
            break;
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_to_text_basic() {
        let html = "<html><body><h1>Hello World</h1><p>This is a test.</p></body></html>";
        let text = html_to_text(html);
        assert!(text.contains("Hello World"));
        assert!(text.contains("This is a test."));
        assert!(!text.contains("<html>"));
    }

    #[test]
    fn test_html_to_text_strips_scripts() {
        let html = r#"<html><body><script>alert('xss')</script><p>Visible</p></body></html>"#;
        let text = html_to_text(html);
        assert!(text.contains("Visible"));
        assert!(!text.contains("alert('xss')"));
    }

    #[test]
    fn test_html_to_text_normalizes_spaces() {
        let html = "<p>Hello    World</p>";
        let text = html_to_text(html);
        assert!(text.contains("Hello World"));
    }

    #[test]
    fn test_html_entity_decoding() {
        let html = "<p>Alice &amp; Bob &lt; test</p>";
        let text = html_to_text(html);
        assert!(text.contains("Alice & Bob"));
        assert!(text.contains("< test"));
    }

    #[test]
    fn test_normalize_whitespace() {
        let input = "Line 1\n\n  Line 2  with  spaces  \nLine 3";
        let output = normalize_whitespace(input);
        assert!(output.contains("Line 1"));
        assert!(output.contains("Line 2 with spaces"));
        assert!(output.contains("Line 3"));
        assert_eq!(output.lines().count(), 3);
    }

    #[test]
    fn test_extract_ddg_results() {
        // Minimal DDG Lite HTML structure
        let html = r#"
<a class="result__a" href="https://example.com"><b>Example</b> Title</a>
<a class="result__snippet">This is a snippet description.</a>
<a class="result__a" href="https://test.com">Test Result</a>
"#;
        let results = extract_ddg_results(html);
        assert!(!results.is_empty());
        assert!(results.iter().any(|(t, _, _)| t.contains("Example") || t.contains("Title")));
    }
}
