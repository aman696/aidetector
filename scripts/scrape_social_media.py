#!/usr/bin/env python3
"""
Social Media Screenshot Scraper for AI Image Detection Dataset.

Takes screenshots of posts from known AI-art and real-photography accounts
on Instagram, X/Twitter, Reddit, and YouTube using your logged-in Firefox
browser session.

Usage:
    python scripts/scrape_social_media.py --all
    python scripts/scrape_social_media.py --platform instagram
    python scripts/scrape_social_media.py --platform reddit --max 20
    python scripts/scrape_social_media.py --platform x --headless

The script copies your Firefox profile cookies to a temp directory,
so you do NOT need to close Firefox first (though closing it is safer
if you run into cookie issues).

Output:
    data/socialmediaai_ss/    ← AI-generated image screenshots
    data/socialmediareal_ss/  ← Real photograph screenshots
"""

import os
import sys
import time
import random
import shutil
import tempfile
import argparse
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
)

try:
    from webdriver_manager.firefox import GeckoDriverManager
    HAS_WDM = True
except ImportError:
    HAS_WDM = False

# ── Paths ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scrape_config import AI_ACCOUNTS, REAL_ACCOUNTS, SCRAPER_SETTINGS


# ── Helpers ─────────────────────────────────────────────────────────────

def _delay(lo=1.5, hi=3.0):
    """Human-like random delay."""
    time.sleep(random.uniform(lo, hi))


def _safe_name(s: str) -> str:
    """Sanitise a string for use in filenames."""
    return s.replace("/", "_").replace("@", "").replace(".", "_").replace(" ", "_")


# ── Scraper ─────────────────────────────────────────────────────────────

class SocialMediaScraper:
    def __init__(self, max_per_account: int = 15, headless: bool = False):
        self.max_per = max_per_account
        self.headless = headless
        self.driver = None
        self.tmp_profile = None

        # Output dirs
        self.ai_dir = PROJECT_ROOT / "data" / "socialmediaai_ss"
        self.real_dir = PROJECT_ROOT / "data" / "socialmediareal_ss"
        self.ai_dir.mkdir(parents=True, exist_ok=True)
        self.real_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {"ai": 0, "real": 0, "skipped": 0, "errors": 0}

    # ── output helpers ──────────────────────────────────────────────────

    def _out_dir(self, label: str) -> Path:
        return self.ai_dir if label == "ai" else self.real_dir

    def _save_screenshot(self, label: str, platform: str,
                         account: str, idx: int,
                         element=None) -> Path:
        """Save a screenshot of a specific element (or viewport fallback).

        If *element* is given, screenshots just that element so we capture
        only the image content — no browser chrome, sidebars, or UI.
        Falls back to viewport screenshot if element screenshot fails.
        """
        name = f"{platform}_{_safe_name(account)}_{idx:03d}.png"
        fp = self._out_dir(label) / name
        saved = False
        if element is not None:
            try:
                element.screenshot(str(fp))
                saved = True
            except Exception:
                pass  # fall through to viewport
        if not saved:
            self.driver.save_screenshot(str(fp))
        self.stats[label] += 1
        return fp

    def _find_largest_img(self, container=None, min_size: int = 150):
        """Find the largest <img> element (by rendered area) in a container.

        Skips tiny icons/avatars (< min_size px on either axis).
        Returns the element or None.
        """
        root = container or self.driver
        best, best_area = None, 0
        for img in root.find_elements(By.TAG_NAME, "img"):
            try:
                w = img.size.get("width", 0)
                h = img.size.get("height", 0)
                if w < min_size or h < min_size:
                    continue
                area = w * h
                if area > best_area:
                    best, best_area = img, area
            except StaleElementReferenceException:
                continue
        return best

    # ── browser lifecycle ───────────────────────────────────────────────

    def start_browser(self):
        """Launch Firefox with a copy of the user's profile (for cookies)."""
        src = Path(SCRAPER_SETTINGS["firefox_profile"]).expanduser()
        if not src.exists():
            print(f"ERROR: Firefox profile not found: {src}")
            print("Edit SCRAPER_SETTINGS['firefox_profile'] in scrape_config.py")
            sys.exit(1)

        print(f"  Copying Firefox cookies from {src.name}...")
        self.tmp_profile = Path(tempfile.mkdtemp(prefix="ff_scrape_"))

        # Copy only auth-relevant files (fast, avoids GBs of cache)
        for item in [
            "cookies.sqlite", "cookies.sqlite-wal", "cookies.sqlite-shm",
            "cert9.db", "key4.db", "pkcs11.txt",
            "logins.json", "logins-backup.json",
            "permissions.sqlite", "signedInUser.json",
            "storage",  # localStorage — some sites keep auth tokens here
        ]:
            s = src / item
            d = self.tmp_profile / item
            if s.exists():
                if s.is_dir():
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

        opts = Options()
        opts.profile = str(self.tmp_profile)
        if self.headless:
            opts.add_argument("--headless")
        opts.add_argument("--width=1920")
        opts.add_argument("--height=1080")
        # Suppress notification / push popups
        opts.set_preference("dom.webnotifications.enabled", False)
        opts.set_preference("dom.push.enabled", False)

        if HAS_WDM:
            svc = Service(GeckoDriverManager().install())
        else:
            svc = Service()  # expects geckodriver on PATH

        print("  Starting Firefox...")
        self.driver = webdriver.Firefox(service=svc, options=opts)
        self.driver.set_page_load_timeout(
            SCRAPER_SETTINGS["page_load_timeout_sec"]
        )
        print("  Browser ready.\n")

    def stop_browser(self):
        if self.driver:
            self.driver.quit()
            self.driver = None
        if self.tmp_profile and self.tmp_profile.exists():
            shutil.rmtree(self.tmp_profile, ignore_errors=True)
            self.tmp_profile = None

    # ── generic scrolling / popup helpers ───────────────────────────────

    def _scroll(self, px: int = 800):
        self.driver.execute_script(f"window.scrollBy(0, {px});")
        _delay(1.0, 1.8)

    def _scroll_to_top(self):
        self.driver.execute_script("window.scrollTo(0, 0);")
        _delay(0.5, 1.0)

    def _click_if_found(self, by, value):
        """Click an element if it exists, silently ignore otherwise."""
        try:
            el = self.driver.find_element(by, value)
            el.click()
            _delay(0.4, 0.8)
            return True
        except (NoSuchElementException, WebDriverException):
            return False

    # ── INSTAGRAM ───────────────────────────────────────────────────────

    def scrape_instagram(self, accounts: list, label: str):
        self._section_header("INSTAGRAM", label)
        for acct in accounts:
            self._ig_account(acct, label)

    def _ig_dismiss_popups(self):
        for txt in ["Allow all cookies", "Accept All",
                     "Allow essential and optional cookies"]:
            self._click_if_found(
                By.XPATH, f"//button[contains(text(), '{txt}')]"
            )
        for txt in ["Not Now", "Not now", "Decline"]:
            self._click_if_found(
                By.XPATH, f"//button[contains(text(), '{txt}')]"
            )

    def _ig_account(self, username: str, label: str):
        url = f"https://www.instagram.com/{username}/"
        print(f"\n  @{username}: loading profile...")
        try:
            self.driver.get(url)
            _delay(3.0, 2.0)
            self._ig_dismiss_popups()

            # Check for 404 / private
            src = self.driver.page_source[:3000].lower()
            if "page not found" in src or "sorry, this page" in src:
                print(f"    SKIP — account not found")
                self.stats["skipped"] += 1
                return
            if "this account is private" in src:
                print(f"    SKIP — account is private")
                self.stats["skipped"] += 1
                return

            # Scroll to load posts, then collect post links
            for _ in range(5):
                self._scroll(600)
            self._scroll_to_top()

            links = set()
            for a in self.driver.find_elements(By.TAG_NAME, "a"):
                try:
                    href = a.get_attribute("href") or ""
                    if "/p/" in href or "/reel/" in href:
                        links.add(href)
                except StaleElementReferenceException:
                    continue

            if not links:
                print(f"    SKIP — no posts found")
                self.stats["skipped"] += 1
                return

            links = list(links)
            take = min(len(links), self.max_per)
            print(f"    Found {len(links)} posts → screenshotting {take}")

            count = 0
            for i, link in enumerate(links[:take]):
                try:
                    self.driver.get(link)
                    _delay(2.0, 1.5)
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "img"))
                    )
                    _delay(1.0, 0.8)
                    img_el = self._find_largest_img(min_size=200)
                    if img_el is None:
                        self.stats["skipped"] += 1
                        continue
                    fp = self._save_screenshot(
                        label, "ig", username, i + 1, element=img_el
                    )
                    count += 1
                    print(f"    [{count}/{take}] {fp.name}")
                except Exception as e:
                    self.stats["errors"] += 1
                    print(f"    error on post {i+1}: {e}")

            print(f"    ✓ {count} screenshots from @{username}")

        except Exception as e:
            self.stats["errors"] += 1
            print(f"    ERROR: {e}")

    # ── X / TWITTER ─────────────────────────────────────────────────────

    def scrape_x(self, accounts: list, label: str):
        self._section_header("X / TWITTER", label)
        for acct in accounts:
            self._x_account(acct, label)

    def _x_account(self, username: str, label: str):
        # /media tab filters to tweets containing images/videos
        url = f"https://x.com/{username}/media"
        print(f"\n  @{username}: loading media tab...")
        try:
            self.driver.get(url)
            _delay(3.5, 2.0)

            src = self.driver.page_source[:3000].lower()
            if "this account doesn" in src or "doesn't exist" in src:
                print(f"    SKIP — account not found")
                self.stats["skipped"] += 1
                return
            if "these tweets are protected" in src:
                print(f"    SKIP — account is protected")
                self.stats["skipped"] += 1
                return

            # Scroll to load tweets
            for _ in range(4):
                self._scroll(900)
            self._scroll_to_top()

            # Collect tweet links
            links = set()
            for a in self.driver.find_elements(By.TAG_NAME, "a"):
                try:
                    href = a.get_attribute("href") or ""
                    if "/status/" in href and username.lower() in href.lower():
                        base = href.split("/photo")[0].split("/video")[0]
                        links.add(base)
                except StaleElementReferenceException:
                    continue

            if not links:
                print(f"    SKIP — no media tweets found")
                self.stats["skipped"] += 1
                return

            links = list(links)
            take = min(len(links), self.max_per)
            print(f"    Found {len(links)} media tweets → screenshotting {take}")

            count = 0
            for i, link in enumerate(links[:take]):
                try:
                    self.driver.get(link)
                    _delay(2.5, 1.5)
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, 'article[data-testid="tweet"]')
                        )
                    )
                    _delay(1.0, 0.8)
                    # Find the image inside the tweet article
                    tweet = self.driver.find_element(
                        By.CSS_SELECTOR, 'article[data-testid="tweet"]'
                    )
                    img_el = self._find_largest_img(
                        container=tweet, min_size=150
                    )
                    if img_el is None:
                        self.stats["skipped"] += 1
                        continue
                    fp = self._save_screenshot(
                        label, "x", username, i + 1, element=img_el
                    )
                    count += 1
                    print(f"    [{count}/{take}] {fp.name}")
                except Exception as e:
                    self.stats["errors"] += 1
                    print(f"    error on tweet {i+1}: {e}")

            print(f"    ✓ {count} screenshots from @{username}")

        except Exception as e:
            self.stats["errors"] += 1
            print(f"    ERROR: {e}")

    # ── REDDIT ──────────────────────────────────────────────────────────

    def scrape_reddit(self, subreddits: list, label: str):
        self._section_header("REDDIT", label)
        for sub in subreddits:
            self._reddit_sub(sub, label)

    def _reddit_dismiss_popups(self):
        # Cookie consent
        for sel in [
            "//button[contains(text(), 'Accept all')]",
            "//button[contains(text(), 'Accept')]",
        ]:
            self._click_if_found(By.XPATH, sel)
        # Close login modal
        self._click_if_found(By.CSS_SELECTOR, "button[aria-label='Close']")
        # Reddit app upsell
        for sel in [
            "//button[contains(text(), 'Continue')]",
            "//a[contains(text(), 'Continue')]",
            "//button[contains(text(), 'Not now')]",
        ]:
            self._click_if_found(By.XPATH, sel)

    def _reddit_sub(self, subreddit: str, label: str):
        sub = subreddit.replace("r/", "").strip("/")
        url = f"https://www.reddit.com/r/{sub}/"
        print(f"\n  r/{sub}: loading...")
        try:
            self.driver.get(url)
            _delay(3.0, 2.0)
            self._reddit_dismiss_popups()
            _delay(1.0, 0.5)

            # Scroll to load posts
            for _ in range(5):
                self._scroll(700)
            self._scroll_to_top()

            # Collect post links
            links = set()
            for a in self.driver.find_elements(By.TAG_NAME, "a"):
                try:
                    href = a.get_attribute("href") or ""
                    if f"/r/{sub}/comments/" in href.lower():
                        base = href.split("?")[0].split("#")[0]
                        links.add(base)
                except StaleElementReferenceException:
                    continue

            if not links:
                print(f"    SKIP — no posts found")
                self.stats["skipped"] += 1
                return

            links = list(links)
            take = min(len(links), self.max_per)
            print(f"    Found {len(links)} posts → screenshotting {take}")

            count = 0
            for i, link in enumerate(links[:take]):
                try:
                    self.driver.get(link)
                    _delay(2.5, 1.5)
                    self._reddit_dismiss_popups()

                    # Wait for any image to appear
                    try:
                        WebDriverWait(self.driver, 8).until(
                            EC.presence_of_element_located((By.TAG_NAME, "img"))
                        )
                    except TimeoutException:
                        # Text-only post — skip
                        self.stats["skipped"] += 1
                        continue

                    _delay(1.5, 1.0)
                    img_el = self._find_largest_img(min_size=200)
                    if img_el is None:
                        self.stats["skipped"] += 1
                        continue
                    fp = self._save_screenshot(
                        label, "reddit", sub, i + 1, element=img_el
                    )
                    count += 1
                    print(f"    [{count}/{take}] {fp.name}")
                except Exception as e:
                    self.stats["errors"] += 1
                    print(f"    error on post {i+1}: {e}")

            print(f"    ✓ {count} screenshots from r/{sub}")

        except Exception as e:
            self.stats["errors"] += 1
            print(f"    ERROR: {e}")

    # ── YOUTUBE ─────────────────────────────────────────────────────────

    def scrape_youtube(self, channels: list, label: str):
        self._section_header("YOUTUBE", label)
        for ch in channels:
            self._yt_channel(ch, label)

    def _yt_channel(self, channel: str, label: str):
        handle = channel.lstrip("@")
        url = f"https://www.youtube.com/@{handle}/community"
        print(f"\n  @{handle}: loading community tab...")
        try:
            self.driver.get(url)
            _delay(3.0, 2.0)

            # Cookie consent
            self._click_if_found(
                By.XPATH, "//button[contains(., 'Accept all')]"
            )
            self._click_if_found(
                By.XPATH, "//button[contains(., 'Reject all')]"
            )

            # Scroll
            for _ in range(3):
                self._scroll(600)

            posts = self.driver.find_elements(
                By.CSS_SELECTOR, "ytd-backstage-post-thread-renderer"
            )

            if not posts:
                print(f"    No community posts — trying video thumbnails...")
                self._yt_thumbnails(handle, label)
                return

            take = min(len(posts), self.max_per)
            print(f"    Found {len(posts)} community posts → screenshotting {take}")

            count = 0
            for i, post in enumerate(posts[:take]):
                try:
                    img_el = self._find_largest_img(
                        container=post, min_size=150
                    )
                    if img_el is None:
                        continue
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block:'center'});",
                        img_el,
                    )
                    _delay(1.0, 0.8)
                    fp = self._save_screenshot(
                        label, "yt", handle, i + 1, element=img_el
                    )
                    count += 1
                    print(f"    [{count}/{take}] {fp.name}")
                except Exception as e:
                    self.stats["errors"] += 1

            print(f"    ✓ {count} screenshots from @{handle}")

        except Exception as e:
            self.stats["errors"] += 1
            print(f"    ERROR: {e}")

    def _yt_thumbnails(self, handle: str, label: str):
        """Fallback: screenshot video pages (shows thumbnail)."""
        url = f"https://www.youtube.com/@{handle}/videos"
        try:
            self.driver.get(url)
            _delay(3.0, 1.5)

            links = set()
            for a in self.driver.find_elements(By.TAG_NAME, "a"):
                try:
                    href = a.get_attribute("href") or ""
                    if "/watch?v=" in href:
                        links.add(href.split("&")[0])
                except StaleElementReferenceException:
                    continue

            if not links:
                print(f"    No videos found")
                return

            links = list(links)
            take = min(len(links), self.max_per)
            print(f"    Found {len(links)} videos → screenshotting {take}")

            count = 0
            for i, link in enumerate(links[:take]):
                try:
                    self.driver.get(link)
                    _delay(2.5, 1.5)
                    # Pause video and grab the video/thumbnail element
                    try:
                        vid = self.driver.find_element(By.TAG_NAME, "video")
                        self.driver.execute_script(
                            "arguments[0].pause();", vid
                        )
                    except Exception:
                        pass
                    img_el = self._find_largest_img(min_size=200)
                    fp = self._save_screenshot(
                        label, "yt", handle, i + 1, element=img_el
                    )
                    count += 1
                    print(f"    [{count}/{take}] {fp.name}")
                except Exception as e:
                    self.stats["errors"] += 1

            print(f"    ✓ {count} thumbnail screenshots from @{handle}")
        except Exception as e:
            self.stats["errors"] += 1
            print(f"    ERROR: {e}")

    # ── orchestration ───────────────────────────────────────────────────

    def _section_header(self, platform: str, label: str):
        tag = "AI" if label == "ai" else "REAL"
        print(f"\n{'=' * 55}")
        print(f"  {platform} — {tag} accounts")
        print(f"{'=' * 55}")

    def run(self, platforms: list = None):
        all_platforms = ["instagram", "x", "reddit", "youtube"]
        platforms = platforms or all_platforms

        try:
            self.start_browser()

            for plat in platforms:
                ai_list = AI_ACCOUNTS.get(plat, [])
                real_list = REAL_ACCOUNTS.get(plat, [])

                scrape_fn = {
                    "instagram": self.scrape_instagram,
                    "x": self.scrape_x,
                    "reddit": self.scrape_reddit,
                    "youtube": self.scrape_youtube,
                }.get(plat)

                if not scrape_fn:
                    continue
                if ai_list:
                    scrape_fn(ai_list, "ai")
                if real_list:
                    scrape_fn(real_list, "real")

            self._print_summary()
        finally:
            self.stop_browser()

    def _print_summary(self):
        print(f"\n{'=' * 55}")
        print("  SCRAPING COMPLETE")
        print(f"{'=' * 55}")
        print(f"  New AI screenshots:   {self.stats['ai']}")
        print(f"  New Real screenshots: {self.stats['real']}")
        print(f"  Skipped:              {self.stats['skipped']}")
        print(f"  Errors:               {self.stats['errors']}")

        ai_total = len(list(self.ai_dir.glob("*.png")))
        real_total = len(list(self.real_dir.glob("*.png")))
        print(f"\n  Total in socialmediaai_ss/:   {ai_total}")
        print(f"  Total in socialmediareal_ss/: {real_total}")
        print(f"\n  AI dir:   {self.ai_dir}")
        print(f"  Real dir: {self.real_dir}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape social media screenshots for AI detection dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/scrape_social_media.py --all
  python scripts/scrape_social_media.py --platform instagram
  python scripts/scrape_social_media.py --platform reddit --max 20
  python scripts/scrape_social_media.py --platform x --headless
        """,
    )
    parser.add_argument(
        "--platform", "-p",
        choices=["instagram", "x", "reddit", "youtube"],
        help="Scrape a specific platform",
    )
    parser.add_argument(
        "--all", "-a", action="store_true",
        help="Scrape all platforms",
    )
    parser.add_argument(
        "--max", "-m", type=int,
        default=SCRAPER_SETTINGS["max_screenshots_per_account"],
        help="Max screenshots per account (default: %(default)s)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run browser without visible window",
    )

    args = parser.parse_args()

    if not args.all and not args.platform:
        parser.print_help()
        print("\nSpecify --all or --platform <name>")
        sys.exit(1)

    print("=" * 55)
    print("  Social Media Screenshot Scraper")
    print("  for AI Image Detection Dataset")
    print("=" * 55)
    print()
    print("  Tip: Close Firefox first for cleanest cookie copy.")
    print("  Edit scripts/scrape_config.py to change accounts.\n")

    platforms = [args.platform] if args.platform else None
    scraper = SocialMediaScraper(
        max_per_account=args.max,
        headless=args.headless,
    )
    scraper.run(platforms=platforms)


if __name__ == "__main__":
    main()
