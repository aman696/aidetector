"""
generate_real_screenshots.py

Captures real desktop/browser screenshots from a curated list of websites
to use as "Real" training/test data for the screenshot classifier.

Usage:
    python scripts/generate_real_screenshots.py
    python scripts/generate_real_screenshots.py --count 100 --out data/screenshots
    python scripts/generate_real_screenshots.py --count 50  --out data/real_test
    python scripts/generate_real_screenshots.py --humans     # sites that show real human faces
    python scripts/generate_real_screenshots.py --mix        # blend UI + human sites evenly

Options:
    --out     DIR    Output directory (default: data/screenshots)
    --count   N      How many screenshots to capture (default: 60)
    --humans         Use sites that prominently show real human faces/photos
    --mix            Blend UI sites and human-photo sites 50/50 (recommended)
    --width   W      Browser viewport width  (default: 1280)
    --height  H      Browser viewport height (default: 800)
    --delay   S      Seconds to wait after page load (default: 2)
    --verbose        Print per-screenshot details

The script cycles through the site list until it hits --count.
Duplicate sites are scrolled to a different position for variety.
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path

# ── UI / text-heavy sites (original list) ────────────────────────────────────
SITES = [
    # News / text-heavy
    "https://en.wikipedia.org/wiki/Special:Random",
    "https://en.wikipedia.org/wiki/Special:Random",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Fast_Fourier_transform",
    "https://news.ycombinator.com",
    "https://news.ycombinator.com/?p=2",
    "https://www.bbc.com/news",
    "https://www.bbc.com/science",
    "https://www.reuters.com",
    "https://apnews.com",
    # Docs / code
    "https://docs.python.org/3/library/numpy.html",
    "https://docs.python.org/3/tutorial/index.html",
    "https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html",
    "https://scikit-learn.org/stable/modules/svm.html",
    "https://opencv.org",
    "https://matplotlib.org/stable/tutorials/index.html",
    "https://developer.mozilla.org/en-US/docs/Web/CSS",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
    # Forums / community
    "https://stackoverflow.com/questions?tab=newest",
    "https://stackoverflow.com/questions?tab=Votes",
    "https://www.reddit.com/r/linux",
    "https://www.reddit.com/r/programming",
    "https://www.reddit.com/r/MachineLearning",
    # GitHub
    "https://github.com/trending",
    "https://github.com/numpy/numpy",
    "https://github.com/scikit-learn/scikit-learn",
    "https://github.com/AUTOMATIC1111/stable-diffusion-webui",
    # Blogs / tutorials
    "https://towardsdatascience.com",
    "https://medium.com/topic/artificial-intelligence",
    "https://realpython.com",
    "https://www.geeksforgeeks.org/machine-learning/",
    "https://paperswithcode.com/latest",
    # Maps / dashboards
    "https://www.openstreetmap.org/#map=12/28.6139/77.2090",
    "https://www.openstreetmap.org/#map=10/51.5074/-0.1278",
    # Project-specific
    "https://arxiv.org/abs/1911.00686",
    "https://arxiv.org/abs/2304.06408",
    "https://arxiv.org/list/cs.CV/recent",
    # Misc content-rich
    "https://www.imdb.com/chart/top",
    "https://www.goodreads.com/list/show/1.Best_Books_Ever",
    "https://fonts.google.com",
    "https://color.adobe.com/explore",
    "https://css-tricks.com",
    "https://www.smashingmagazine.com",
    "https://devdocs.io/python/",
    "https://explainshell.com",
    "https://explain.dalibo.com",
    "https://regex101.com",
    "https://caniuse.com",
]

# ── Human-photo sites — pages displaying REAL human faces & bodies ────────────
# These counter-balance AI-generated faces in the AI screenshot training set.
# All sources use authentic camera/video images, not AI-generated content.
HUMAN_SITES = [
    # Wikipedia pages of real people — encyclopaedic portraits
    "https://en.wikipedia.org/wiki/Barack_Obama",
    "https://en.wikipedia.org/wiki/Elon_Musk",
    "https://en.wikipedia.org/wiki/Malala_Yousafzai",
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://en.wikipedia.org/wiki/Marie_Curie",
    "https://en.wikipedia.org/wiki/Muhammad_Ali",
    "https://en.wikipedia.org/wiki/Cristiano_Ronaldo",
    "https://en.wikipedia.org/wiki/Serena_Williams",
    "https://en.wikipedia.org/wiki/Stephen_Hawking",
    "https://en.wikipedia.org/wiki/Usain_Bolt",
    # Free stock photo sites — real human photography
    "https://unsplash.com/s/photos/portrait",
    "https://unsplash.com/s/photos/people",
    "https://unsplash.com/s/photos/crowd",
    "https://unsplash.com/s/photos/athlete",
    "https://www.pexels.com/search/people/",
    "https://www.pexels.com/search/woman/",
    "https://www.pexels.com/search/man/",
    "https://www.pexels.com/search/crowd/",
    "https://pixabay.com/photos/search/people/",
    "https://pixabay.com/photos/search/portrait/",
    # Wikimedia portrait category
    "https://commons.wikimedia.org/wiki/Category:Portraits",
    "https://commons.wikimedia.org/wiki/Category:Photographs_of_people",
    "https://commons.wikimedia.org/wiki/Category:Portrait_photographs_of_women",
    "https://commons.wikimedia.org/wiki/Category:Portrait_photographs_of_men",
    # Photo journalism / news with faces
    "https://www.reuters.com/news/pictures/",
    "https://apnews.com/hub/photos",
    "https://www.bbc.com/news/in_pictures",
    "https://time.com/section/photography/",
    "https://www.nationalgeographic.com/photography/",
    # Sports (authentic action shots of real humans)
    "https://en.wikipedia.org/wiki/2024_Summer_Olympics",
    "https://en.wikipedia.org/wiki/FIFA_World_Cup",
    "https://www.bbc.com/sport",
    # Film / celebrity biographies (camera photographs, not AI)
    "https://www.imdb.com/name/nm0000375/",   # Robert Downey Jr.
    "https://www.imdb.com/name/nm1373737/",   # Emma Stone
    "https://www.imdb.com/name/nm0000093/",   # Brad Pitt
    "https://www.imdb.com/name/nm0001401/",   # Cate Blanchett
]

# Scroll offsets to capture different parts of long pages
SCROLL_OFFSETS_PX = [0, 400, 800, 1200, 1800, 2600]


def run(args):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Error: playwright not installed.")
        print("Fix: pip install playwright && python -m playwright install firefox")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose site list based on flags
    if args.humans:
        active_sites = HUMAN_SITES
        mode_label   = "human-photo"
    elif args.mix:
        # interleave both lists
        active_sites = []
        for a, b in zip(SITES, HUMAN_SITES):
            active_sites.extend([a, b])
        active_sites += SITES[len(HUMAN_SITES):] + HUMAN_SITES[len(SITES):]
        mode_label   = "UI + human-photo (mixed)"
    else:
        active_sites = SITES
        mode_label   = "UI/text"

    print(f"Site mode : {mode_label}  ({len(active_sites)} unique URLs)")

    # Expand site list by cycling if needed
    site_pool = []
    for scroll in SCROLL_OFFSETS_PX:
        for url in active_sites:
            site_pool.append((url, scroll))
    random.shuffle(site_pool)


    if len(site_pool) < args.count:
        # repeat with different shuffles
        extra = site_pool[:]
        while len(site_pool) < args.count:
            random.shuffle(extra)
            site_pool += extra
    site_pool = site_pool[: args.count]

    print(f"Generating {args.count} real screenshots → {out_dir}/")
    print(f"Viewport: {args.width}×{args.height}")
    print()

    saved   = 0
    skipped = 0
    start   = time.time()

    with sync_playwright() as pw:
        browser = pw.firefox.launch(headless=True)
        context = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
                "Gecko/20100101 Firefox/120.0"
            ),
        )
        page = context.new_page()
        # Block heavy media to speed things up
        page.route("**/*.{mp4,webm,ogg,mp3,wav}", lambda r: r.abort())

        for idx, (url, scroll) in enumerate(site_pool, 1):
            fname = out_dir / f"real_web_{idx:04d}.png"

            try:
                page.goto(url, timeout=20_000, wait_until="domcontentloaded")
                time.sleep(args.delay)

                if scroll > 0:
                    page.evaluate(f"window.scrollTo(0, {scroll})")
                    time.sleep(0.4)

                page.screenshot(path=str(fname), full_page=False)

                saved += 1
                if args.verbose:
                    print(f"  [{idx:3d}/{args.count}] {os.path.basename(fname)}"
                          f"  scroll={scroll}px  {url[:60]}")
                else:
                    # Progress bar
                    bar_len = 40
                    filled  = int(bar_len * saved / args.count)
                    bar     = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r  [{bar}] {saved}/{args.count}", end="", flush=True)

            except Exception as e:
                skipped += 1
                if args.verbose:
                    print(f"  [{idx:3d}/{args.count}] SKIP — {e}")
                continue

        browser.close()

    elapsed = time.time() - start
    print()
    print(f"\nDone. Saved {saved} screenshots in {elapsed:.0f}s  (skipped {skipped})")
    print(f"Output directory: {out_dir.resolve()}")
    print()
    print("Next steps:")
    print(f"  1. Move any bad screenshots out of {out_dir}/")
    print(f"  2. python main.py --train-screenshot")


def main():
    parser = argparse.ArgumentParser(
        description="Generate real browser screenshots for AI-detector training data."
    )
    parser.add_argument("--out",     default="data/screenshots",
                        help="Output directory (default: data/screenshots)")
    parser.add_argument("--count",   type=int, default=60,
                        help="Number of screenshots to capture (default: 60)")
    parser.add_argument("--humans",  action="store_true",
                        help="Use sites that prominently show real human faces")
    parser.add_argument("--mix",     action="store_true",
                        help="Blend UI sites and human-photo sites 50/50")
    parser.add_argument("--width",   type=int, default=1280,
                        help="Browser viewport width  (default: 1280)")
    parser.add_argument("--height",  type=int, default=800,
                        help="Browser viewport height (default: 800)")
    parser.add_argument("--delay",   type=float, default=2.0,
                        help="Seconds to wait after page load (default: 2)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-screenshot details instead of progress bar")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
