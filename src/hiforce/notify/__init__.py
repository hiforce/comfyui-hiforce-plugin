from abc import abstractmethod


class ImageSaveNotify:
    @abstractmethod
    def notify(self, images):
        pass
