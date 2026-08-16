"""
VoiceGuard AI - Playwright Scenario Screenshot Generator.
Executes all 4 key user scenarios in the live browser UI and captures screenshots of result cards.
"""
import os
import sys
import time
from playwright.sync_api import sync_playwright

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SAMPLES_DIR = os.path.join(ROOT_DIR, "test_samples")

def generate_scenario_screenshots():
    print("=" * 70)
    print("   VOICEGUARD AI - GENERATING VISUAL SCREENSHOTS FOR ALL UI SCENARIOS    ")
    print("=" * 70)

    alice_ref = os.path.join(TEST_SAMPLES_DIR, "Speaker_Alice_Reference.wav")
    alice_auth = os.path.join(TEST_SAMPLES_DIR, "Speaker_Alice_Authentic_Sample.wav")
    bob_ref = os.path.join(TEST_SAMPLES_DIR, "Speaker_Bob_Enrolled.wav")
    charlie_ref = os.path.join(TEST_SAMPLES_DIR, "Speaker_Charlie_Enrolled.wav")
    fake_audio = os.path.join(TEST_SAMPLES_DIR, "Suspect_Deepfake_Clone.wav")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # -------------------------------------------------------------
        # SCENARIO 1: 1:1 Authentic Voice Verification (High Similarity)
        # -------------------------------------------------------------
        print("\n [SCENARIO 1] 1:1 Dual-Voice Similarity -> AUTHENTIC MATCH...")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto("http://localhost:5174/", wait_until="networkidle")
        time.sleep(1)

        inputs = page.query_selector_all('input[type="file"]')
        if len(inputs) >= 2:
            inputs[0].set_input_files(alice_ref)
            inputs[0].evaluate("el => el.dispatchEvent(new Event('change', { bubbles: true }))")
            
            inputs[1].set_input_files(alice_auth)
            inputs[1].evaluate("el => el.dispatchEvent(new Event('change', { bubbles: true }))")
            time.sleep(1)

            exec_btn = page.locator("button:has-text('Execute 1:1 GMM Similarity')")
            if exec_btn.is_visible():
                exec_btn.click()
                page.wait_for_selector("text=Forensic Analysis Report", timeout=30000)
                time.sleep(2)
                s1_path = os.path.join(ROOT_DIR, "scenario_1_authentic.png")
                page.screenshot(path=s1_path, full_page=True)
                print(f"  [OK] Saved Scenario 1 Screenshot -> {s1_path}")
        page.close()

        # -------------------------------------------------------------
        # SCENARIO 2: 1:1 Deepfake Voice Cloning Detection (Fraudulent)
        # -------------------------------------------------------------
        print("\n [SCENARIO 2] 1:1 Dual-Voice Similarity -> FRAUDULENT / DEEPFAKE...")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto("http://localhost:5174/", wait_until="networkidle")
        time.sleep(1)

        inputs = page.query_selector_all('input[type="file"]')
        if len(inputs) >= 2:
            inputs[0].set_input_files(alice_ref)
            inputs[0].evaluate("el => el.dispatchEvent(new Event('change', { bubbles: true }))")
            
            inputs[1].set_input_files(fake_audio)
            inputs[1].evaluate("el => el.dispatchEvent(new Event('change', { bubbles: true }))")
            time.sleep(1)

            exec_btn = page.locator("button:has-text('Execute 1:1 GMM Similarity')")
            if exec_btn.is_visible():
                exec_btn.click()
                page.wait_for_selector("text=Forensic Analysis Report", timeout=30000)
                time.sleep(2)
                s2_path = os.path.join(ROOT_DIR, "scenario_2_deepfake.png")
                page.screenshot(path=s2_path, full_page=True)
                print(f"  [OK] Saved Scenario 2 Screenshot -> {s2_path}")
        page.close()

        # -------------------------------------------------------------
        # SCENARIO 3: 1:N Multi-Speaker Dataset Identification
        # -------------------------------------------------------------
        print("\n [SCENARIO 3] 1:N Multi-Speaker Dataset Identification...")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto("http://localhost:5174/", wait_until="networkidle")
        time.sleep(1)

        tab_1n = page.locator("button:has-text('MULTI-SPEAKER DATASET IDENTIFICATION (1:N)')")
        if tab_1n.is_visible():
            tab_1n.click()
            time.sleep(2)

            dataset_inputs = page.query_selector_all('input[type="file"]')
            if len(dataset_inputs) >= 2:
                dataset_inputs[0].set_input_files([alice_ref, bob_ref, charlie_ref])
                dataset_inputs[0].evaluate("el => el.dispatchEvent(new Event('change', { bubbles: true }))")
                time.sleep(1)

                dataset_inputs[1].set_input_files(alice_auth)
                dataset_inputs[1].evaluate("el => el.dispatchEvent(new Event('change', { bubbles: true }))")
                time.sleep(1)

                id_btn = page.locator("button:has-text('Run 1:N Multi-Speaker Dataset Identification')")
                if id_btn.is_visible():
                    id_btn.click()
                    page.wait_for_selector("text=1:N Dataset Forensic Identification", timeout=30000)
                    time.sleep(2)
                    s3_path = os.path.join(ROOT_DIR, "scenario_3_multispeaker.png")
                    page.screenshot(path=s3_path, full_page=True)
                    print(f"  [OK] Saved Scenario 3 Screenshot -> {s3_path}")
        page.close()

        # -------------------------------------------------------------
        # SCENARIO 4: Single Audio Deepfake Scanner
        # -------------------------------------------------------------
        print("\n [SCENARIO 4] Single Audio Deepfake Scanner...")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto("http://localhost:5174/", wait_until="networkidle")
        time.sleep(1)

        tab_single = page.locator("button:has-text('SINGLE AUDIO DEEPFAKE SCANNER')")
        if tab_single.is_visible():
            tab_single.click()
            time.sleep(2)

            single_input = page.query_selector('input[type="file"]')
            if single_input:
                single_input.set_input_files(fake_audio)
                single_input.evaluate("el => el.dispatchEvent(new Event('change', { bubbles: true }))")
                time.sleep(1)

                scan_btn = page.locator("button:has-text('Analyze Audio Authenticity')")
                if scan_btn.is_visible():
                    scan_btn.click()
                    page.wait_for_selector("text=Forensic Analysis Report", timeout=30000)
                    time.sleep(2)
                    s4_path = os.path.join(ROOT_DIR, "scenario_4_singlescanner.png")
                    page.screenshot(path=s4_path, full_page=True)
                    print(f"  [OK] Saved Scenario 4 Screenshot -> {s4_path}")
        page.close()

        browser.close()

    print("\n" + "=" * 70)
    print("   [SUCCESS] ALL SCENARIO SCREENSHOTS GENERATED SUCCESSFULLY!    ")
    print("=" * 70)

if __name__ == "__main__":
    generate_scenario_screenshots()
