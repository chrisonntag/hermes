from abc import ABC, abstractmethod


class DataSource(ABC):
    @abstractmethod
    def get_documents(self):
        pass

    @abstractmethod
    def get_name(self):
        pass


