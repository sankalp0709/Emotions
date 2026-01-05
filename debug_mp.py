import mediapipe as mp
print("dir(mp):", dir(mp))
try:
    import mediapipe.python.solutions as solutions
    print("Found solutions in mediapipe.python.solutions")
except ImportError:
    print("Could not find mediapipe.python.solutions")

try:
    print("mp.solutions:", mp.solutions)
except AttributeError as e:
    print("Error accessing mp.solutions:", e)
