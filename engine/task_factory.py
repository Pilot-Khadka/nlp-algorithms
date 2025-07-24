from tasks.language_modeling import LanguageModelingTask
from tasks.sentiment_analysis import SentimentAnalysisTask
# from tasks.translation import TranslationTask


def load_task(task_name):
    if task_name == "language_modeling":
        return LanguageModelingTask()
    elif task_name == "sentiment_analysis":
        return SentimentAnalysisTask()
    else:
        raise ValueError(f"Unknown task: {task_name}")
