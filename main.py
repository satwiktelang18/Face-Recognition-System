import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
import tkinter as tk
from tkinter import filedialog

# ---------------- CONFIG ----------------
DATABASE_PATH = "database"
THRESHOLD = 0.4   # Cosine similarity (higher = more strict)

# ---------------- INIT MODEL ----------------
print("🚀 Initializing model...")
app = FaceAnalysis()
app.prepare(ctx_id=-1)  # CPU safe

# ---------------- LOAD DATABASE ----------------
known_embeddings = []
known_names = []

print("🔄 Loading face database...")

if not os.path.exists(DATABASE_PATH):
    print("❌ 'database' folder not found!")
    exit()

for person_name in os.listdir(DATABASE_PATH):
    person_folder = os.path.join(DATABASE_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    for file in os.listdir(person_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(person_folder, file)

            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Cannot read {img_path}")
                continue

            faces = app.get(img)

            if len(faces) > 0:
                known_embeddings.append(faces[0].embedding)
                known_names.append(person_name)
            else:
                print(f"⚠️ No face in {img_path}")

# Convert to numpy
if len(known_embeddings) == 0:
    print("\n❌ No valid faces found in database!")
    print("👉 Make sure:")
    print("   - Images contain clear faces")
    print("   - Folder structure is correct")
    exit()

known_embeddings = np.array(known_embeddings)

print("✅ Database loaded successfully!\n")

# ---------------- INPUT IMAGE ----------------
root = tk.Tk()
root.withdraw()

print("📂 Opening file browser...")
file_path = filedialog.askopenfilename(
    title="Select an image to test",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
)

if not file_path:
    print("❌ No file selected!")
    exit()

print(f"✅ Selected: {file_path}")

if not os.path.exists(file_path):
    print("❌ File not found!")
    exit()

img = cv2.imread(file_path)

if img is None:
    print("❌ Failed to load image!")
    exit()

# ---------------- RECOGNITION ----------------
faces = app.get(img)

recognized = 0

for face in faces:
    emb = face.embedding

    # ✅ Cosine similarity (correct method for buffalo_l)
    emb_norm = emb / np.linalg.norm(emb)
    db_norms = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)

    similarities = np.dot(db_norms, emb_norm)
    best_match = np.argmax(similarities)

    print(f"🔍 Best match: {known_names[best_match]} | Similarity: {similarities[best_match]:.4f}")

    if similarities[best_match] > THRESHOLD:
        name = known_names[best_match]
        recognized += 1
    else:
        name = "Unknown"

    # Draw bounding box
    x1, y1, x2, y2 = face.bbox.astype(int)

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(img, name, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

# ---------------- OUTPUT ----------------
if len(faces) == 0:
    print("❌ No faces detected")
elif recognized == 0:
    print("⚠️ Faces detected but not recognized")
else:
    print("✅ Faces recognized")

cv2.imwrite("result.jpg", img)
print("📁 Result saved as result.jpg")
print("👉 Open result.jpg to see the output!")