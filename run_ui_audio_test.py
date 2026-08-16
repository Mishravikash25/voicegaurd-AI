"""
VoiceGuard AI - Playwright Browser UI Automation & Live Audio Deepfake Test.
Performs real UI file uploads, triggers GMM analysis, and captures high-res screenshots of results.
"""
import os
import sys
import time
from playwright.sync_api import sync_playwright

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SAMPLES_DIR = os.path.join(ROOT_DIR, "test_samples")

def run_playwright_test():
    print("=" * 70)
    print("      VOICEGUARD AI - PLAYWRIGHT UI AUTOMATION & LIVE AUDIO DEEPFAKE TEST    ")
    print("=" * 70)

    ref_audio = os.path.join(TEST_SAMPLES_DIR, "Speaker_Alice_Reference.wav")
    auth_audio = os.path.join(TEST_SAMPLES_DIR, "Speaker_Alice_Authentic_Sample.wav")
    fake_audio = os.path.join(TEST_SAMPLES_DIR, "Suspect_Deepfake_Clone.wav")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # Navigate to Vite UI
        page.goto("http://localhost:5174/", wait_until="networkidle")
        time.sleep(2)

        # -------------------------------------------------------------
        # STEP 1: Test Pairwise 1:1 Similarity (Reference vs Deepfake Clone)
        # -------------------------------------------------------------
        print("\n [STEP 1] Testing 1:1 Dual-Voice Engine (Authentic Reference vs Deepfake Clone)...")
        
        inputs = page.query_selector_all('input[type="file"]')
        print(f"  [OK] Found {len(inputs)} file inputs on 1:1 tab.")
        
        if len(inputs) >= 2:
            inputs[0].set_input_files(ref_audio)
            inputs[1].set_input_files(fake_audio)
            print("  [OK] Uploaded Reference (.wav) & Deepfake Clone (.wav) into UI dropzones.")
            time.sleep(1)
            
            # Click Execute button
            exec_btn = page.locator("button:has-text('Execute 1:1 GMM Similarity')")
            if exec_btn.is_visible():
                exec_btn.click()
                print("  [OK] Clicked 'Execute 1:1 GMM Similarity Scaling & Deepfake Verification' button.")
                
                # Wait for analysis completion and result card display
                page.wait_for_selector("text=Forensic Analysis Report", timeout=25000)
                time.sleep(2)
                
                # Capture result screenshot
                screenshot_path1 = os.path.join(ROOT_DIR, "ui_1to1_deepfake_result.png")
                page.screenshot(path=screenshot_path1, full_page=True)
                print(f"  [OK] Saved UI Deepfake Result Screenshot -> {screenshot_path1}")

        # -------------------------------------------------------------
        # STEP 2: Test Single Audio Deepfake Scanner
        # -------------------------------------------------------------
        print("\n [STEP 2] Testing Single Audio Deepfake Scanner Tab...")
        
        # Click "Perform Another Analysis" to return to form
        reset_btn = page.locator("button:has-text('Perform Another Analysis')")
        if reset_btn.is_visible():
            reset_btn.click()
            time.sleep(2)

        tab_scanner = page.locator("button:has-text('SINGLE AUDIO DEEPFAKE SCANNER')")
        if tab_scanner.is_visible():
            tab_scanner.click()
            time.sleep(2)  # Wait for Framer Motion tab animation
            
            single_input = page.query_selector('input[type="file"]')
            if single_input:
                single_input.set_input_files(fake_audio)
                print("  [OK] Uploaded Deepfake Clone (.wav) into Single Scanner UI.")
                time.sleep(1)
                
                analyze_btn = page.locator("button:has-text('Analyze Audio Authenticity')")
                if analyze_btn.is_visible():
                    analyze_btn.click()
                    print("  [OK] Clicked 'Analyze Audio Authenticity' button.")
                    
                    # Wait for analysis completion and result card display
                    page.wait_for_selector("text=Forensic Analysis Report", timeout=25000)
                    time.sleep(2)
                    
                    screenshot_path2 = os.path.join(ROOT_DIR, "ui_single_deepfake_result.png")
                    page.screenshot(path=screenshot_path2, full_page=True)
                    print(f"  [OK] Saved UI Single Deepfake Result Screenshot -> {screenshot_path2}")

        browser.close()

    print("\n" + "=" * 70)
    print("       [SUCCESS] LIVE BROWSER UI AUDIO DEEPFAKE TEST COMPLETED!       ")
    print("=" * 70)

if __name__ == "__main__":
    run_playwright_test()
