import mediapipe
print("Path:", mediapipe.__path__)
try:
    import mediapipe.python as mpp
    print("mediapipe.python dir:", dir(mpp))
except ImportError as e:
    print("Import mediapipe.python failed:", e)

try:
    from mediapipe import solutions
    print("from mediapipe import solutions works")
except ImportError as e:
    print("from mediapipe import solutions failed:", e)
