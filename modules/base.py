from abc import ABC, abstractmethod


class DataSource(ABC):
    @abstractmethod
    def extract_documents(self):
        pass

    @abstractmethod
    def get_name(self):
        pass


