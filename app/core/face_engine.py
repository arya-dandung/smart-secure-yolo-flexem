import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, folder_path="known_faces"):
        print("🧠 [CPU] Memuat Model Wajah (InsightFace)...")
        
        # providers=['CPUExecutionProvider'] memaksa ONNXRuntime pakai CPU
        self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
        self.known_embeddings = []
        self.known_names = []
        self.similarity_threshold = 0.5  # Sesuaikan (0.4 - 0.6)
        
        # Load database saat inisialisasi
        self.load_database(folder_path)

    def load_database(self, folder_path):
        # Cek apakah folder ada, jika tidak buat baru
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"⚠️ Folder '{folder_path}' tidak ditemukan. Dibuat baru.")
            return

        print(f"📂 Membaca Wajah dari: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(folder_path, filename)
                name = os.path.splitext(filename)[0] # arya.jpg -> arya
                
                img = cv2.imread(path)
                if img is None: continue

                faces = self.app.get(img)
                if len(faces) > 0:
                    self.known_embeddings.append(faces[0].embedding)
                    self.known_names.append(name)
                    print(f"   ✅ Terdaftar: {name}")
                else:
                    print(f"   ⚠️ Wajah tidak terbaca: {filename}")
        
        print(f"✨ Total {len(self.known_names)} wajah di database.")

    def recognize_crop(self, face_img_bgr):
        """Mengenali siapa pemilik wajah dari potongan gambar"""
        faces = self.app.get(face_img_bgr)
        
        if len(faces) == 0:
            return "Unknown"
        
        target_emb = faces[0].embedding
        max_score = 0
        best_name = "Unknown"

        for idx, known_emb in enumerate(self.known_embeddings):
            # Hitung kemiripan (Cosine Similarity)
            score = np.dot(target_emb, known_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(known_emb))
            
            if score > max_score:
                max_score = score
                if score > self.similarity_threshold:
                    best_name = self.known_names[idx]
        
        return best_name
