from ultralytics import YOLO

model = YOLO("yolo11s.pt")

orang_unik = set()
kendaraan_unik = set()
puncak = 0

for hasil in model.track(
    source="14806026_2160_3840_32fps.mp4",
    device="mps",
    classes=[0, 2, 3, 5, 7],
    conf=0.4,
    persist=True,
    stream=True,
    verbose=False,
):
    if hasil.boxes.id is None:
        continue
    ids = hasil.boxes.id.int().tolist()
    kelas = hasil.boxes.cls.int().tolist()

    sekarang = 0
    for i, k in zip(ids, kelas):
        if k == 0:
            orang_unik.add(i)
            sekarang += 1
        else:
            kendaraan_unik.add(i)
    puncak = max(puncak, sekarang)

print(f"Orang unik: {len(orang_unik)}")
print(f"Kendaraan unik: {len(kendaraan_unik)}")
print(f"Puncak orang bersamaan: {puncak}")