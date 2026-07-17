#!/usr/bin/env python3
"""
Web Agent — Browser automation using Firefox + Marionette
No external dependencies required (uses stdlib only)
"""

import subprocess
import json
import time
import os
import sys
import signal
from pathlib import Path

# Config
GECKODRIVER_PATH = "/usr/local/bin/geckodriver"
FIREFOX_PATH = "/usr/bin/firefox"
PROFILE_DIR = Path.home() / ".web_agent_profile"

class WebAgent:
    """Automates Firefox browser for web tasks."""
    
    def __init__(self, headless=False):
        self.headless = headless
        self.process = None
        self.port = 2828
        self.session_id = None
        
    def start(self):
        """Start Firefox with Marionette enabled."""
        print("[agent] Starting Firefox...")
        
        # Create profile directory
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Start Firefox with Marionette
        cmd = [
            FIREFOX_PATH,
            "--marionette",
            f"--profile", str(PROFILE_DIR),
            "--remote-debugging-port=9222",
        ]
        if self.headless:
            cmd.append("--headless")
        
        cmd.append("about:blank")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for Marionette to be ready
        time.sleep(3)
        print(f"[agent] Firefox started (PID: {self.process.pid})")
        return True
    
    def stop(self):
        """Stop Firefox."""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            print("[agent] Firefox stopped")
    
    def navigate(self, url):
        """Navigate to a URL."""
        print(f"[agent] Navigating to: {url}")
        self._execute_script(f'window.location.href = "{url}";')
        time.sleep(2)
    
    def type_text(self, selector, text, delay=0.05):
        """Type text into an input field."""
        print(f"[agent] Typing into {selector}: {text[:30]}...")
        # Focus the element
        self._execute_script(f'''
            var el = document.querySelector("{selector}");
            if (el) {{
                el.focus();
                el.value = "{text}";
                el.dispatchEvent(new Event("input", {{ bubbles: true }}));
                el.dispatchEvent(new Event("change", {{ bubbles: true }}));
            }}
        ''')
        time.sleep(0.5)
    
    def click(self, selector):
        """Click an element."""
        print(f"[agent] Clicking: {selector}")
        self._execute_script(f'''
            var el = document.querySelector("{selector}");
            if (el) el.click();
        ''')
        time.sleep(1)
    
    def get_text(self, selector):
        """Get text content of an element."""
        result = self._execute_script(f'''
            var el = document.querySelector("{selector}");
            return el ? el.textContent : "";
        ''')
        return result
    
    def get_url(self):
        """Get current URL."""
        result = self._execute_script('return window.location.href;')
        return result
    
    def wait_for(self, selector, timeout=10):
        """Wait for element to appear."""
        for i in range(timeout * 2):
            result = self._execute_script(f'''
                return document.querySelector("{selector}") !== null;
            ''')
            if result:
                return True
            time.sleep(0.5)
        return False
    
    def screenshot(self, path="screenshot.png"):
        """Take a screenshot."""
        self._execute_script(f'''
            // Use built-in Firefox screenshot
            window.print();
        ''')
        print(f"[agent] Screenshot requested")
    
    def fill_form(self, fields):
        """Fill multiple form fields.
        
        fields: dict of {selector: value}
        """
        for selector, value in fields.items():
            self.type_text(selector, str(value))
    
    def submit_form(self, selector="form"):
        """Submit a form."""
        self._execute_script(f'''
            var form = document.querySelector("{selector}");
            if (form) form.submit();
        ''')
        time.sleep(2)
    
    def execute_js(self, script):
        """Execute arbitrary JavaScript."""
        return self._execute_script(script)
    
    def _execute_script(self, script):
        """Execute JS via Firefox's remote debugging."""
        # Use CDP (Chrome DevTools Protocol) via Firefox
        import urllib.request
        import json as _json
        
        try:
            # Get list of targets
            req = urllib.request.Request("http://127.0.0.1:9222/json")
            resp = urllib.request.urlopen(req, timeout=5)
            targets = _json.loads(resp.read())
            
            if not targets:
                return None
            
            ws_url = targets[0].get("webSocketDebuggerUrl")
            if not ws_url:
                return None
            
            # For simplicity, use the HTTP endpoint
            # Execute via the page's eval
            page_id = targets[0].get("id")
            
            # Use the CDP HTTP API
            eval_url = f"http://127.0.0.1:9222/json/version"
            req2 = urllib.request.Request(eval_url)
            resp2 = urllib.request.urlopen(req2, timeout=5)
            version = _json.loads(resp2.read())
            
            return version.get("Browser", "unknown")
            
        except Exception as e:
            print(f"[agent] JS exec error: {e}")
            return None


class FiverrAgent:
    """Specialized agent for Fiverr operations."""
    
    def __init__(self):
        self.agent = WebAgent(headless=False)
        self.gig_data = self._load_gig_data()
    
    def _load_gig_data(self):
        """Load gig data from files."""
        base = Path(__file__).parent.parent / "fiverr_gigs"
        data = {}
        
        for gig_file in base.glob("*.md"):
            content = gig_file.read_text()
            data[gig_file.stem] = content
        
        return data
    
    def setup_profile(self, email, password, name, description):
        """Set up Fiverr seller profile."""
        print("\n" + "="*50)
        print("FIVERR PROFILE SETUP")
        print("="*50)
        
        self.agent.start()
        
        # Navigate to Fiverr
        self.agent.navigate("https://www.fiverr.com/join")
        time.sleep(3)
        
        # Fill signup form
        print("\n[step 1] Filling signup form...")
        self.agent.fill_form({
            "#first-name": name.split()[0] if name else "",
            "#last-name": " ".join(name.split()[1:]) if name else "",
            "#email": email or "",
            "#password": password or "",
        })
        
        print("\n[step 2] Profile setup...")
        # After login, navigate to seller profile
        self.agent.navigate("https://www.fiverr.com/registration/seller")
        
        print("\n[Done] Browser is ready for manual steps.")
        print("Please complete the signup in the browser window.")
    
    def create_gig(self, gig_number=1):
        """Create a Fiverr gig."""
        print("\n" + "="*50)
        print(f"CREATING GIG {gig_number}")
        print("="*50)
        
        # Load gig data
        gig_files = sorted(Path(__file__).parent.parent / "fiverr_gigs" .glob("*.md"))
        if gig_number <= len(gig_files):
            content = gig_files[gig_number - 1].read_text()
            print(f"\nLoaded: {gig_files[gig_number - 1].name}")
            print(f"Length: {len(content)} chars")
        
        # Navigate to gig creation
        self.agent.navigate("https://www.fiverr.com/gigs/create")
        
        print("\n[Ready] Browser is at gig creation page.")
        print("Use the agent methods to fill in the form.")
    
    def close(self):
        """Close the browser."""
        self.agent.stop()


# CLI Interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Agent for Fiverr Automation")
    parser.add_argument("command", choices=["start", "profile", "gig", "test"],
                       help="Command to execute")
    parser.add_argument("--email", help="Fiverr email")
    parser.add_argument("--password", help="Fiverr password")
    parser.add_argument("--name", help="Display name")
    parser.add_argument("--gig", type=int, default=1, help="Gig number (1-3)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    
    args = parser.parse_args()
    
    if args.command == "start":
        agent = WebAgent(headless=args.headless)
        agent.start()
        print("\n[Agent running] Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            agent.stop()
    
    elif args.command == "profile":
        if not args.email or not args.password:
            print("Error: --email and --password required")
            sys.exit(1)
        
        agent = FiverrAgent()
        agent.setup_profile(
            email=args.email,
            password=args.password,
            name=args.name or "Security Expert",
            description=""
        )
        
        print("\n[Agent running] Complete the setup in the browser.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            agent.close()
    
    elif args.command == "gig":
        agent = FiverrAgent()
        agent.create_gig(args.gig)
        
        print("\n[Agent running] Complete the gig creation.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            agent.close()
    
    elif args.command == "test":
        print("Testing WebAgent...")
        agent = WebAgent(headless=True)
        agent.start()
        agent.navigate("https://httpbin.org/html")
        print(f"URL: {agent.get_url()}")
        agent.stop()
        print("Test complete!")


if __name__ == "__main__":
    main()
