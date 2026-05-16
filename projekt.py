from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_CSV = DATA_DIR / "sign_mnist_train.csv"
TEST_CSV = DATA_DIR / "sign_mnist_test.csv"
MODEL_PATH = BASE_DIR / "sign_mnist_live_model.keras"
FACE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

LABEL_VALUES = [
	0,
	1,
	2,
	3,
	4,
	5,
	6,
	7,
	8,
	10,
	11,
	12,
	13,
	14,
	15,
	16,
	17,
	18,
	19,
	20,
	21,
	22,
	23,
	24,
]
LABEL_TO_CLASS = {label: index for index, label in enumerate(LABEL_VALUES)}
CLASS_TO_LABEL = {index: label for label, index in LABEL_TO_CLASS.items()}
CLASS_NAMES = [
	"A",
	"B",
	"C",
	"D",
	"E",
	"F",
	"G",
	"H",
	"I",
	"K",
	"L",
	"M",
	"N",
	"O",
	"P",
	"Q",
	"R",
	"S",
	"T",
	"U",
	"V",
	"W",
	"X",
	"Y",
]


def set_seeds(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


def load_sign_mnist(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
	features: list[list[float]] = []
	labels: list[int] = []

	with csv_path.open(newline="") as file_handle:
		reader = csv.DictReader(file_handle)
		for row in reader:
			label_value = int(row["label"])
			if label_value not in LABEL_TO_CLASS:
				continue

			pixels = [float(row[f"pixel{index}"]) / 255.0 for index in range(1, 785)]
			features.append(pixels)
			labels.append(LABEL_TO_CLASS[label_value])

	x_data = np.asarray(features, dtype=np.float32).reshape(-1, 28, 28, 1)
	y_data = np.asarray(labels, dtype=np.int32)
	return x_data, y_data


def build_model() -> tf.keras.Model:
	model = tf.keras.Sequential(
		[
			tf.keras.layers.Input(shape=(28, 28, 1)),
			tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
			tf.keras.layers.MaxPooling2D(),
			tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
			tf.keras.layers.MaxPooling2D(),
			tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
			tf.keras.layers.GlobalAveragePooling2D(),
			tf.keras.layers.Dense(128, activation="relu"),
			tf.keras.layers.Dropout(0.35),
			tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
		]
	)

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model


def train_model(model_path: Path = MODEL_PATH) -> tf.keras.Model:
	if not TRAIN_CSV.exists():
		raise FileNotFoundError(f"Nie znaleziono pliku treningowego: {TRAIN_CSV}")

	print("Wczytywanie danych treningowych...")
	x_train, y_train = load_sign_mnist(TRAIN_CSV)

	x_val = None
	y_val = None
	if TEST_CSV.exists():
		print("Wczytywanie danych testowych...")
		x_val, y_val = load_sign_mnist(TEST_CSV)

	model = build_model()
	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			monitor="val_accuracy" if x_val is not None else "loss",
			patience=3,
			restore_best_weights=True,
		),
		tf.keras.callbacks.ReduceLROnPlateau(
			monitor="val_loss" if x_val is not None else "loss",
			patience=2,
			factor=0.5,
			verbose=1,
		),
	]

	print("Trenowanie modelu...")
	model.fit(
		x_train,
		y_train,
		validation_data=(x_val, y_val) if x_val is not None else None,
		epochs=12,
		batch_size=128,
		callbacks=callbacks,
		verbose=2,
	)

	if x_val is not None and y_val is not None:
		loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
		print(f"Dokładność na zbiorze testowym: {accuracy:.4f}")

	model.save(model_path)
	print(f"Model zapisany do: {model_path}")
	return model


def load_or_train_model(force_retrain: bool = False) -> tf.keras.Model:
	if MODEL_PATH.exists() and not force_retrain:
		print(f"Ładowanie gotowego modelu: {MODEL_PATH}")
		return tf.keras.models.load_model(MODEL_PATH)

	return train_model(MODEL_PATH)


def preprocess_hand_image(image: np.ndarray) -> np.ndarray:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	gray = cv2.equalizeHist(gray)

	if float(np.mean(gray)) > 127.0:
		gray = cv2.bitwise_not(gray)

	resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
	normalized = resized.astype(np.float32) / 255.0
	return normalized.reshape(28, 28, 1)


def detect_face_regions(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
	face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
	return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def overlaps_face(box: tuple[int, int, int, int], faces: list[tuple[int, int, int, int]]) -> bool:
	x1, y1, x2, y2 = box
	for face_x, face_y, face_w, face_h in faces:
		face_x2 = face_x + face_w
		face_y2 = face_y + face_h
		intersect_x1 = max(x1, face_x)
		intersect_y1 = max(y1, face_y)
		intersect_x2 = min(x2, face_x2)
		intersect_y2 = min(y2, face_y2)
		if intersect_x2 > intersect_x1 and intersect_y2 > intersect_y1:
			return True
	return False


def extract_hand_roi(frame: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
	height, width = frame.shape[:2]
	faces = detect_face_regions(frame)
	ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	skin_mask_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
	skin_mask_hsv = cv2.inRange(hsv, (0, 30, 60), (25, 200, 255))
	skin_mask = cv2.bitwise_or(skin_mask_ycrcb, skin_mask_hsv)

	kernel = np.ones((5, 5), np.uint8)
	skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
	skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
	skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)

	contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)

	min_area = (height * width) * 0.03
	for contour in contours:
		area = cv2.contourArea(contour)
		if area < min_area:
			continue

		x, y, box_width, box_height = cv2.boundingRect(contour)
		if y + box_height < int(height * 0.35):
			continue

		padding = int(max(box_width, box_height) * 0.2)
		x1 = max(0, x - padding)
		y1 = max(0, y - padding)
		x2 = min(width, x + box_width + padding)
		y2 = min(height, y + box_height + padding)
		candidate_box = (x1, y1, x2, y2)

		if overlaps_face(candidate_box, faces):
			continue

		roi = frame[y1:y2, x1:x2]
		if roi.size > 0:
			return roi, candidate_box

	return None, None


def prepare_live_sample(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None]:
	mirrored = cv2.flip(frame, 1)
	roi, bounding_box = extract_hand_roi(mirrored)
	if roi is None:
		return mirrored, None, None
	sample = preprocess_hand_image(roi)
	return mirrored, sample, bounding_box


def class_name_from_prediction(class_index: int) -> str:
	label_value = CLASS_TO_LABEL[class_index]
	label_position = LABEL_VALUES.index(label_value)
	return CLASS_NAMES[label_position]


def draw_overlay(frame: np.ndarray, label: str, confidence: float) -> np.ndarray:
	overlay = frame.copy()
	text = f"{label}  {confidence * 100:.1f}%"
	cv2.rectangle(overlay, (15, 15), (390, 90), (0, 0, 0), -1)
	cv2.putText(
		overlay,
		"Rozpoznany gest",
		(25, 42),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		(255, 255, 255),
		2,
		cv2.LINE_AA,
	)
	cv2.putText(
		overlay,
		text,
		(25, 75),
		cv2.FONT_HERSHEY_SIMPLEX,
		1.0,
		(0, 220, 0),
		2,
		cv2.LINE_AA,
	)
	return overlay


def run_live_camera(model: tf.keras.Model, camera_index: int = 0) -> None:
	capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
	if not capture.isOpened():
		capture.release()
		capture = cv2.VideoCapture(camera_index)

	if not capture.isOpened():
		raise RuntimeError("Nie udało się otworzyć kamery. Sprawdź, czy laptopowa kamera jest dostępna.")

	print("Sterowanie: q - wyjście, r - odśwież model z pliku, spacja - pauza")
	paused = False
	last_prediction = ("-", 0.0)

	while True:
		if not paused:
			ok, frame = capture.read()
			if not ok:
				print("Nie udało się pobrać klatki z kamery.")
				break

			mirrored, sample, bounding_box = prepare_live_sample(frame)
			if sample is None:
				last_prediction = ("Brak dłoni", 0.0)
				preview = draw_overlay(mirrored, "Brak dłoni", 0.0)
				cv2.imshow("Rozpoznawanie gestow ASL", preview)
			else:
				probabilities = model.predict(sample[np.newaxis, ...], verbose=0)[0]
				class_index = int(np.argmax(probabilities))
				confidence = float(probabilities[class_index])
				gesture = class_name_from_prediction(class_index)
				last_prediction = (gesture, confidence)

				preview = mirrored
				if bounding_box is not None:
					x1, y1, x2, y2 = bounding_box
					cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)

				preview = draw_overlay(preview, gesture, confidence)
				cv2.imshow("Rozpoznawanie gestow ASL", preview)
		else:
			cv2.imshow("Rozpoznawanie gestow ASL", np.zeros((480, 640, 3), dtype=np.uint8))

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
		if key == ord(" "):
			paused = not paused
		if key == ord("r"):
			print("Przeładowywanie modelu z dysku...")
			model = tf.keras.models.load_model(MODEL_PATH)

	capture.release()
	cv2.destroyAllWindows()
	print(f"Ostatni wynik: {last_prediction[0]} ({last_prediction[1] * 100:.1f}%)")


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Rozpoznawanie gestów z języka migowego z laptopowej kamery na podstawie Sign MNIST."
	)
	parser.add_argument("--retrain", action="store_true", help="Wymuś ponowne trenowanie modelu.")
	parser.add_argument("--camera", type=int, default=0, help="Indeks kamery (domyślnie 0).")
	return parser


def main() -> None:
	os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
	set_seeds()

	parser = build_argument_parser()
	args = parser.parse_args()

	model = load_or_train_model(force_retrain=args.retrain)
	run_live_camera(model, camera_index=args.camera)


if __name__ == "__main__":
	main()
