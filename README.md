# EasyCheck — Working two-device MVP (October 2026)

**Purpose:** Demonstrate a four-card, two-device classroom assessment system. A phone scans cards, a teacher's laptop chooses a question and displays results in the same website. Only **one Output** and **one Scanner** role can be connected simultaneously. No student accounts, paid AI service or cloud database are required.

**What is implemented:**

- Four actual printable 150 x 70 mm EasyCheck cards: 20 mm orange header, 2 mm orange edges, ArUco identifier, 4 writable white MCQ bubbles (A/B/C/D), white answer area.
- Multiple cards per photo, different positions/rotations, marker identification, perspective correction, independent bubble-centre refinement, MCQ mark detection and automatic grading.
- Written-number recognition for the math example using **offline Tesseract OCR + a small digit classifier** trained on a real handwritten-digit dataset. It falls back to a visible suggestion and **Teacher Review** when uncertain. It does not claim to reliably understand arbitrary handwriting, fractions, free-text reasoning or full-page working.
- Two linked browser roles with a six-character pairing code. Teacher chooses between **Math: `2 + 2 = ?` (written answer `4`)** and **Science: Red Planet (MCQ answer `C`)**. Phone automatically changes scan mode; laptop updates roughly within two seconds.
- Per-question results and summary, per-card crop viewing, teacher corrections, reset that invalidates both devices, emergency PIN, photo upload, live camera capture on HTTPS, and a **Synthetic test scan** button using the actual computer vision code.
- Single-process memory-only demo: no user or student identity records, no database, and no persistent storage.

## Tonight: test the entire website without physical cards

On your **output device**, open the website and click **Use this device as OUTPUT**. Choose **Math** or **MCQ**; a six-character pairing code appears.

On your **phone**, open the **same URL**. Enter that pairing code and click **Use this device as SCANNER**. The current question appears.

Click **Synthetic test scan** on the phone. You can also use the included `samples/MCQ-example.jpg` and `samples/Math-example.jpg` files in the normal **Choose image > Scan selected photo** flow to test image uploads without taking a new picture. It generates four fictional cards and sends that generated photo through the **same actual image-processing pipeline** as a real upload. The output screen should automatically display 4 results. Try both questions, view each cropped answer, manually change a response, and click **Reset All** to confirm both devices return to the role selector.

This is a software integration test, **not a substitute for tomorrow's paper-and-pencil test**.

## Tomorrow: physical cards

1. Use the included `EasyCheck-4-cards.pdf` (or the website's **Download 4-card PDF** button). **Print at 100% / actual size**, NOT "Fit to page." Three cards are on A4 page 1; one on page 2.
2. Cut out four cards along the orange edge, glue them flat to cardboard. Do not cover the markers with tape or glossy film. **Do not print your Aseprite sketch as-is:** the generated marker geometry and circles must match this software. You can embellish the backing without changing the front.
3. For MCQ mode, shade **inside** one white circle (an X inside also works in tests); do not draw rings around the outside. Erase previous choices before the next question.
4. For the math question, write **one large final answer** in the main white area: preferably `4`. Do work in the notebook. The current OCR does not reliably identify full equations, fractions or paragraphs.
5. Place the cards on a table (wood/white/dark background all supported in principle), not overlapping, within the phone frame. Use **good lighting**, avoid motion blur, and keep the ArUco marker plus all four orange edges in view.
6. On the phone, click **Take a photo / choose an image**, then **Scan selected photo**. The scanner shows detected IDs; the output screen shows the grades. A few unreadable cards can be rescanned or manually corrected.

## Easiest public temporary hosting: Render free Web Service

The project contains a `Dockerfile` and `render.yaml` for the [Render free tier](https://render.com/docs/free). It includes the Tesseract OCR executable automatically: no external paid OCR API or account is used.

1. Create a **new GitHub repository**. Upload **the contents of this directory** into the repository root: `server.py`, `vision.py`, `ocr.py`, `cards.py`, `Dockerfile`, `requirements.txt`, `render.yaml`, `static/`, and `assets/`. Do **not** upload only the ZIP as a single file.
2. In [Render](https://dashboard.render.com), create a new **Web Service** from that repository. Choose **Docker** as the runtime and **Free** for the instance plan. Render detects the `Dockerfile`. Alternatively, choose **New > Blueprint** to use the `render.yaml` file.
3. If using the Blueprint, provide a **private emergency reset PIN** when asked for `EASYCHECK_RESET_PIN`. If using a normal Web Service, add this variable in Render's environment settings, or the local demo default of `042826` is used (do not use that known default for a publicly shared site).
4. Wait for the Render deploy to finish. You get an HTTPS URL such as `https://your-app-name.onrender.com`. Open the **same URL on both devices**. HTTPS enables the phone's live camera. The native "Take a photo" button is often even more reliable than continuous live video.
5. Visit `/health` once to verify the web service is ready: expected `{"status":"ok","vision":"aruco","ocr":"tesseract"}`.

**Important on Render Free:** The service sleeps after around 15 minutes without requests, and its temporary memory is lost on restarts or redeploys. When actively open, both devices poll the server, keeping it awake. **Open the site well before the demo**, keep the output tab open, and reset/re-pair if the host restarts. Free hosting cannot give production-level uptime guarantees. Run one Uvicorn worker because both devices share one in-memory state.

If Render reports insufficient memory, use a smaller JPEG photo with four cards; the backend already caps processing to about 10 megapixels and a 3600-pixel longest edge to reduce memory usage.

## Alternative: run on your Arch Linux laptop

In your fish shell, from the extracted `easycheck-live` directory:

```fish
python -m venv .venv
.venv/bin/pip install -r requirements.txt
sudo pacman -S --needed tesseract tesseract-data-eng
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 5000 --workers 1
```

Open `http://127.0.0.1:5000` from the laptop. The phone can upload a photo on the same Wi-Fi through `http://YOUR_LAPTOP_IP:5000`, but **browser live camera usually requires HTTPS**. Use Render for two-device public HTTPS access.

### Diagnostic command (fish, shows output and copies to clipboard)

```fish
.venv/bin/python -m pytest -q 2>&1 | tee /tmp/easycheck-tests.txt
wl-copy < /tmp/easycheck-tests.txt
```

`pytest` is required for the optional local test suite; install it using `.venv/bin/pip install pytest` if necessary. The deployment server itself does not require pytest.

## Source layout

| File | Purpose |
| --- | --- |
| `server.py` | Two-device web server, pairing, live results, questions, reset, corrections |
| `vision.py` | ArUco identification, perspective correction, MCQ mark reading, synthetic camera photos |
| `ocr.py` | Constrained handwritten-number recognition + uncertainty handling |
| `cards.py` | Card dimensions, images and exact-size PDF export |
| `assets/digits_svm.xml` | Small secondary digit classifier (trained offline, included) |
| `static/index.html`, `static/app.js` | Simple browser interface for phone and laptop |
| `tests/test_end_to_end.py` | End-to-end and vision regression tests |
| `Dockerfile`, `render.yaml` | Free-host deployment |
| `train_digit_model.py` | Optional offline model retraining (requires scikit-learn) |

## Reliability boundaries

The automated tests cover photo uploads, four unique printed cards, simultaneous MCQ choices, blank/double-mark review behaviour, JPEG compression, rotated photographs, reduced brightness, both role sessions, pairing/reset, teacher overrides and question changes. The **true reliability of actual pencil handwriting in classroom lighting has not been established**. Tesseract alone is not a state-of-the-art handwritten math interpreter, and the small SVM model trained on the scikit-learn/UCI optical handwritten digits dataset is only a supporting check. Any ambiguous reading must go to teacher review rather than becoming an unverified grade.

This is a public prototype with basic pairing, **not a secure student-information system**. Use anonymous card numbers and avoid uploading actual students' names or confidential answers.
