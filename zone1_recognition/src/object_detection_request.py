class ItemRecognition:
    def __init__ (self, path):
        print("sucess")
        self.path = path
    
    def localize_objects(self):
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()

        with open(self.path, 'rb') as image_file:
            content = image_file.read()
        image = vision.types.Image(content=content)

        objects = client.object_localization(
            image=image).localized_object_annotations

        print('Number of objects found: {}'.format(len(objects)))
        for object_ in objects:
            print('\n{} (confidence: {})'.format(object_.name, object_.score))
            print('Normalized bounding polygon vertices: ')
            for vertex in object_.bounding_poly.normalized_vertices:
                print(' - ({}, {})'.format(vertex.x, vertex.y))
