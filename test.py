import face_recognition as fr
import cv2
import os


def encode_faces(folder):
    list_people_encoding = []
    for filename in os.listdir(folder):
        known_image = fr.load_image_file(f'{folder}{filename}')
        known_encoding = fr.face_encodings(known_image)
        # print(known_encoding)
        try:
            list_people_encoding.append((known_encoding[0], filename))
        except:
            pass
    return list_people_encoding


def person_in_image(file_name):
    n = len(file_name) - 4
    file_name = file_name[:n]
    while True:
        try:
            last_int = int(file_name[n - 1])
            n -= 1
        except:
            break
    return file_name[: n - 1]


def create_frame(location, label, target):
    top, right, bottom, left = location

    cv2.rectangle(target, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.rectangle(target, (left, bottom + 20), (right, bottom), (255, 0, 0), cv2.FILLED)
    cv2.putText(target, label, (left + 3, bottom + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def find_target_face(target_img, target_encod, history, last_frame_detecting, frame_index, seq_length=7000):
    face_location = fr.face_locations(target_img)
    visited = {actor: False for actor in os.listdir("Actors")}
    visited_locations = set()
    for person in people_encoding:
        encoded_face = person[0]
        filename = person[1]
        error_msg = "none"
        try:
            # error_msg = f"encoded_face.shape = {len(encoded_face)} and target_encod.shape = {len(target_encod)}"
            is_target_face = fr.compare_faces(encoded_face, target_encod, tolerance=0.55)
            if face_location:
                face_number = 0
                for location in face_location:
                    if is_target_face[face_number] and location not in visited_locations:
                        label = person_in_image(filename)
                        if not (visited[label]) and frame_index - last_frame_detecting[label] >= seq_length:
                            history[label] += 1
                            face_number += 1
                            visited[label] = True
                            last_frame_detecting[label] = frame_index
                            visited_locations.add(location)
                        create_frame(location, f"{label}: {history[label]}", target_img)

        except ValueError:
            pass

    return history, last_frame_detecting


def render_image(target_img):
    cv2.imshow('Results', target_img)
    # key = cv2.waitKey(1)
    # if key == ord('s'):
    #     output_path = os.path.join(f"Results\\{uuid1()}.jpg")
    #     cv2.imwrite(output_path, target_img)
    # if key == ord(' '):
    #     cv2.destroyAllWindows()


def persons_in_frame(frame, history, last_frame_detecting, frame_index):
    try:
        target_encoding = fr.face_encodings(frame)
    except:
        print(frame)
        return history, last_frame_detecting
    history, last_frame_detecting = find_target_face(frame, target_encoding, history, last_frame_detecting, frame_index)
    render_image(frame)
    return history, last_frame_detecting


people_encoding = encode_faces('peoples\\')


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    hist = {act: 0 for act in os.listdir("Actors")}
    frame_id = 0
    last_frame_detecting = {actor: -10000 for actor in os.listdir("Actors")}
    while cap.isOpened():
        ret, frame = cap.read()
        if frame_id % 5000 == 0:
            hist, last_frame_detecting = persons_in_frame(frame, hist, last_frame_detecting, frame_id)
        frame_id += 1
        if cv2.waitKey(1) == ord(" "):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(hist)

# main("D:\\second year internship\\Summer_Internship_2023\\clean DB\\Saison1 (2005)\\episode_10.mp4")
