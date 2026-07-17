#!/usr/bin/env python3
"""
Simple Web Agent — Uses xdg-open + clipboard for form filling
No dependencies required
"""

import subprocess
import json
import time
import os
import sys
from pathlib import Path

def open_url(url):
    """Open URL in default browser."""
    subprocess.run(["xdg-open", url], check=False)
    print(f"[opened] {url}")

def copy_to_clipboard(text):
    """Copy text to clipboard."""
    # Try xclip first, then xsel
    for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]:
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            p.communicate(text.encode())
            print(f"[copied] {text[:50]}...")
            return True
        except FileNotFoundError:
            continue
    print("[warning] No clipboard tool found. Install xclip or xsel.")
    return False

def paste_from_clipboard():
    """Paste text from clipboard."""
    for cmd in [["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return ""

def type_text(text, delay=0.03):
    """Type text using xdotool."""
    # Copy to clipboard first, then paste
    copy_to_clipboard(text)
    time.sleep(0.1)
    subprocess.run(["xdotool", "key", "ctrl+v"], check=False)
    time.sleep(0.2)

def press_key(key):
    """Press a key using xdotool."""
    subprocess.run(["xdotool", "key", key], check=False)
    time.sleep(0.2)

def click_at(x, y):
    """Click at coordinates using xdotool."""
    subprocess.run(["xdotool", "mousemove", str(x), str(y), "click", "1"], check=False)
    time.sleep(0.2)

def type_into_field(text, tab_count=0):
    """Type into a field (assumes focus is on the field)."""
    if tab_count > 0:
        for _ in range(tab_count):
            press_key("Tab")
            time.sleep(0.1)
    type_text(text)


# ============================================================
# FIVERR AUTOMATION SCRIPT
# ============================================================

FIVERR_GIGS = {
    1: {
        "title": "I will audit your Python code for security vulnerabilities",
        "tags": "python security audit, security vulnerability, code audit, penetration testing, flask security, django security, python review, OWASP, vulnerability assessment, security scan",
        "description": """🔒 PYTHON SECURITY AUDIT — Find and Fix Vulnerabilities Before Attackers Do

Is your Python application secure? I will perform a thorough security audit and deliver a detailed report with actionable fixes.

✅ WHAT YOU GET

• Complete OWASP Top 10 vulnerability analysis
• SQL injection, XSS, CSRF, SSRF, and IDOR detection
• Authentication & authorization review
• Dependency vulnerability check (pip audit)
• Hardcoded secrets & API key detection
• Rate limiting & input validation review
• Detailed PDF report with severity ratings
• Line-by-line fix recommendations
• GitHub issues created for each finding

🛠️ FRAMEWORKS I AUDIT

• Flask, Django, FastAPI
• Bottle, CherryPy, Tornado
• Pure Python scripts & CLI tools
• REST APIs & GraphQL endpoints

📋 WHAT I NEED FROM YOU

• Repository access (GitHub/GitLab) or zip file
• Brief description of the application
• Any specific areas of concern

⚡ DELIVERY

• Basic: 24 hours
• Standard: 48 hours
• Premium: 72 hours

🏆 WHY CHOOSE ME

• I find real vulnerabilities, not just theoretical issues
• Every finding includes a working fix
• Clear, professional reports you can share with your team
• Follow-up support included

💬 Not sure which package is right for you? Message me and I'll help you choose.""",
        "pricing": {
            "basic": {"name": "Security Scan", "price": 25, "desc": "Quick security scan of your Python codebase. Up to 2,000 lines. OWASP Top 10 check. PDF report. 24h delivery."},
            "standard": {"name": "Full Audit", "price": 50, "desc": "Comprehensive security audit. Up to 5,000 lines. Full audit + dependency check. PDF + GitHub issues. 48h delivery."},
            "premium": {"name": "Security Hardening", "price": 100, "desc": "Complete security audit + hardening. Unlimited lines. Full audit + auth review. PDF + issues + remediation plan. 72h delivery."}
        }
    },
    2: {
        "title": "I will fix security vulnerabilities in your Python or JavaScript code",
        "tags": "security fix, python bug fix, XSS fix, SQL injection fix, code security, vulnerability fix, flask fix, django fix, security patch, web security",
        "description": """🐛 SECURITY BUGS IN YOUR CODE? I Will Fix Them Properly

Found vulnerabilities but don't know how to fix them? I fix security bugs the right way — not just patching symptoms, but implementing proper solutions.

🔧 COMMON FIXES I MAKE

• SQL Injection → Parameterized queries & ORM usage
• XSS → Output encoding & Content Security Policy
• CSRF → Token validation & SameSite cookies
• SSRF → URL validation & allowlisting
• IDOR → Authorization checks & object-level permissions
• Path Traversal → Input validation & sandboxing
• Hardcoded Secrets → Environment variables & secret managers
• Weak Crypto → Modern algorithms (bcrypt, argon2)
• Missing Rate Limiting → Token bucket / sliding window
• Insecure Deserialization → Safe alternatives

💻 LANGUAGES & FRAMEWORKS

• Python: Flask, Django, FastAPI, Bottle
• JavaScript: Node.js, Express, React, Next.js
• TypeScript: NestJS, Angular
• Go, Ruby, PHP (contact me)

📦 WHAT YOU GET

• Fixed code ready to deploy
• Explanation of each change
• Tests to prevent regressions
• Security best practices applied
• Clean, maintainable code

⚡ DELIVERY
Same-day for small fixes. 24-48h for full hardening.

💬 Not sure what's wrong? Send me your code and I'll do a free 5-minute assessment.""",
        "pricing": {
            "basic": {"name": "Quick Fix", "price": 30, "desc": "Fix up to 5 security vulnerabilities. Code changes + explanation. 24h delivery."},
            "standard": {"name": "Security Hardening", "price": 75, "desc": "Fix up to 15 security issues. Full hardening + tests. 48h delivery."},
            "premium": {"name": "Full Remediation", "price": 150, "desc": "Unlimited security fixes. Full codebase hardening. Tests + CI/CD config. 72h delivery."}
        }
    },
    3: {
        "title": "I will build you a custom AI automation script or tool",
        "tags": "AI automation, python automation, AI script, ollama, openai, machine learning, workflow automation, data processing, automation bot, custom AI",
        "description": """🤖 CUSTOM AI AUTOMATION — Automate Any Repetitive Task

Need to process documents, analyze data, or automate workflows? I build custom Python scripts using AI (local or cloud) that save you hours every week.

🎯 WHAT I CAN AUTOMATE

• Document processing & data extraction
• Email drafting, classification & responses
• Data entry & form filling
• Report generation & summaries
• Code review & analysis
• Web scraping with AI interpretation
• Image/text analysis & categorization
• Customer support ticket routing
• File organization & naming
• API integration workflows
• Database operations & migrations

🧠 AI OPTIONS

• Local AI (Ollama) — Zero ongoing costs, runs on your machine
• OpenAI GPT-4 — Best quality, ~$0.01-0.10 per task
• Anthropic Claude — Great for long documents
• Gemini — Good for multimodal tasks

🛠️ TECH STACK

• Python 3.10+
• Ollama, OpenAI, Anthropic, Gemini
• LangChain, LlamaIndex
• Selenium, BeautifulSoup (web)
• Pandas, NumPy (data)
• Flask/FastAPI (web interfaces)

📦 WHAT YOU GET

• Custom script tailored to YOUR exact needs
• Clear configuration (no code changes needed)
• Error handling so it doesn't crash
• Setup instructions & documentation
• 7 days of free bug fixes

⚡ DELIVERY
Simple scripts: 24 hours
Complex workflows: 48-72 hours

💬 Not sure if I can automate your task? Message me — if it's possible, I'll tell you how.""",
        "pricing": {
            "basic": {"name": "Simple Script", "price": 20, "desc": "Single-task automation script. Local AI (Ollama). Config + README. 24h delivery."},
            "standard": {"name": "Smart Automation", "price": 50, "desc": "Multi-step automation. Local or cloud AI. Error handling + logging. 48h delivery."},
            "premium": {"name": "Full System", "price": 120, "desc": "Complex workflow automation. Multiple AI models. Web dashboard. 72h delivery."}
        }
    }
}


def setup_fiverr_profile():
    """Interactive Fiverr profile setup."""
    print("\n" + "="*60)
    print("FIVERR PROFILE SETUP")
    print("="*60)
    
    # Open Fiverr
    open_url("https://www.fiverr.com/join")
    input("\n[1] Complete the signup/login in the browser, then press Enter...")
    
    # Open seller profile
    open_url("https://www.fiverr.com/registration/seller")
    input("\n[2] Fill in your seller profile in the browser, then press Enter...")
    
    print("\n[✓] Profile setup complete!")
    return True


def create_gig(gig_number=1):
    """Create a Fiverr gig with guided automation."""
    if gig_number not in FIVERR_GIGS:
        print(f"Invalid gig number: {gig_number}")
        return False
    
    gig = FIVERR_GIGS[gig_number]
    
    print("\n" + "="*60)
    print(f"CREATING GIG {gig_number}: {gig['title'][:50]}...")
    print("="*60)
    
    # Open gig creation page
    open_url("https://www.fiverr.com/gigs/create")
    input("\n[1] The browser is now at gig creation. Press Enter to continue...")
    
    # Step by step guide with clipboard
    print("\n[2] GIG TITLE — Copy this and paste into the title field:")
    print("-" * 50)
    print(gig["title"])
    copy_to_clipboard(gig["title"])
    input("    (Copied to clipboard! Paste with Ctrl+V, then press Enter)")
    
    print("\n[3] TAGS — Copy these and paste into the tags field:")
    print("-" * 50)
    print(gig["tags"])
    copy_to_clipboard(gig["tags"])
    input("    (Copied to clipboard! Paste with Ctrl+V, then press Enter)")
    
    print("\n[4] DESCRIPTION — Copy this and paste into the description field:")
    print("-" * 50)
    print(gig["description"][:200] + "...")
    copy_to_clipboard(gig["description"])
    input("    (Full description copied to clipboard! Paste with Ctrl+V, then press Enter)")
    
    # Pricing
    for tier_name, tier in gig["pricing"].items():
        print(f"\n[5] {tier_name.upper()} PACKAGE:")
        print(f"    Name: {tier['name']}")
        print(f"    Price: ${tier['price']}")
        print(f"    Description: {tier['desc']}")
        copy_to_clipboard(tier["desc"])
        input(f"    (Description copied! Paste into {tier_name} description, then press Enter)")
    
    print("\n[✓] Gig creation steps complete!")
    print("    Review everything in the browser and click Publish.")
    return True


def create_all_gigs():
    """Create all 3 gigs."""
    print("\n" + "="*60)
    print("CREATING ALL 3 FIVERR GIGS")
    print("="*60)
    
    for gig_num in [1, 2, 3]:
        print(f"\n{'='*40}")
        print(f"GIG {gig_num} of 3")
        print(f"{'='*40}")
        create_gig(gig_num)
        
        if gig_num < 3:
            cont = input(f"\nReady to create Gig {gig_num + 1}? (y/n): ")
            if cont.lower() != 'y':
                break
    
    print("\n" + "="*60)
    print("ALL GIGS CREATED!")
    print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fiverr Web Agent")
    parser.add_argument("command", choices=["profile", "gig", "all", "test"],
                       help="Command to run")
    parser.add_argument("--gig", type=int, default=1, choices=[1, 2, 3],
                       help="Gig number (1-3)")
    
    args = parser.parse_args()
    
    if args.command == "profile":
        setup_fiverr_profile()
    elif args.command == "gig":
        create_gig(args.gig)
    elif args.command == "all":
        create_all_gigs()
    elif args.command == "test":
        print("[test] Checking dependencies...")
        for cmd in ["xdg-open", "xdotool", "xclip"]:
            try:
                subprocess.run(["which", cmd], capture_output=True, check=True)
                print(f"  {cmd}: ✓")
            except:
                print(f"  {cmd}: ✗ (install: apt install {cmd})")
        print("[test] Done!")


if __name__ == "__main__":
    main()
