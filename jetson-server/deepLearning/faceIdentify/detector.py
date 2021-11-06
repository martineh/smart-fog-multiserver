import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import os

def load_know_people():
    config_path = os.path.dirname(__file__)

    # Load a sample picture and learn how to recognize it.
    hector_image = face_recognition.load_image_file(config_path + "/database/Hector_Martinez.png")
    hector_face_encoding = face_recognition.face_encodings(hector_image)[0]

    stallone_image = face_recognition.load_image_file(config_path + "/database/Sylvester_Stallone.jpg")
    stallone_face_encoding = face_recognition.face_encodings(stallone_image)[0]

    # Load a second sample picture and learn how to recognize it.
    #biden_image = face_recognition.load_image_file("biden.jpg")
    #biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    
    # Create arrays of known face encodings and their names
    known_face_encodings = [
        hector_face_encoding,
        stallone_face_encoding
    ]
    known_face_names = [
        "Hector Martinez",
        "Sylvester_Stallone"
    ]
    
    return known_face_encodings, known_face_names
    
    

def identify_faces(unknown_image, face_locations):
    known_face_encodings, known_face_names = load_know_people()
    
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)
    names = []
    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        # Remove the drawing library from memory as per the Pillow docs
        
        names.append(name)
        
    del draw
    # Display the resulting image
    pil_image.show()
    # You can also save a copy of the new image to disk if you want by uncommenting this line
    # pil_image.save("image_with_boxes.jpg")
        
    return names