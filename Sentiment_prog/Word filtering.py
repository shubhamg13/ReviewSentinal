camera_file = open(r"C:\Users\nauri\Desktop\ML proj\Camera words.txt", "r")
filter = camera_file.read().split('\n')
Ftext = ["camera is good", "video is nice", "hello this is nice"]
Ftext_camera = []
for l in Ftext:
    for w in filter:
        if w in l and l not in Ftext_camera:
            Ftext_camera.append(l)
print(Ftext_camera)