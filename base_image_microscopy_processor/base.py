from base_processor.imaging import BaseImageProcessor


class BaseMicroscopyImageProcessor(BaseImageProcessor):
    def __init__(self, *args, **kwargs):
        super(BaseMicroscopyImageProcessor, self).__init__(*args, **kwargs)

