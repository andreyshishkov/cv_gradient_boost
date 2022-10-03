class BoundingBox:

    def __init__(self, start_x=0, start_y=0, width=0,
                 height=0, centroid_x=0, centroid_y=0):
        self.start_x = start_x
        self.start_y = start_y
        self.width = width
        self.height = height
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
