# External imports:
from pathlib import Path

# Internal imports:
from base_evaluation import PostgresEvaluation

class GeneralPostgresEvaluation(PostgresEvaluation):
    """General evaluation using predefined benchmark data."""
    def __init__(self, benchmark_data_path: str):
        """
        Initialize with benchmark data path.
        """
        benchmark_path = Path(benchmark_data_path)
        questions_df_path = benchmark_path / 'questions_df.csv'
        corpora_folder_path = benchmark_path / 'corpora'

        if corpora_folder_path.exists():
            corpora_filenames = [f for f in corpora_folder_path.iterdir() if f.is_file()]
            corpora_id_paths = {f.stem: str(f) for f in corpora_filenames}
        else:
            corpora_id_paths = {}

        super().__init__(str(questions_df_path), corpora_id_paths=corpora_id_paths)
        self.is_general = True
        self.benchmark_data_path = benchmark_data_path


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    questions_csv = "path/to/questions.csv"
    corpora_paths = {
        "doc1": "path/to/document1.txt",
        "doc2": "path/to/document2.txt"
    }

    # Initialize evaluation
    evaluator = PostgresEvaluation(questions_csv, corpora_paths)

    # Run single evaluation
    result = evaluator.run(
        chunk_type="recursive",
        chunk_size=400,
        chunk_overlap=100,
        retrieve=5
    )

    print("Evaluation Results:")
    print(f"IoU Mean: {result['iou_mean']:.3f}")
    print(f"Precision Mean: {result['precision_mean']:.3f}")
    print(f"Recall Mean: {result['recall_mean']:.3f}")

    # Run comparative evaluation
    configs = [
        {"chunk_type": "recursive", "chunk_size": 400, "chunk_overlap": 100},
        {"chunk_type": "recursive", "chunk_size": 600, "chunk_overlap": 150},
        {"chunk_type": "token", "chunk_size": 400, "chunk_overlap": 100},
        {"chunk_type": "sentence", "chunk_size": 400, "chunk_overlap": 100}
    ]

    comparison_df = evaluator.run_comparative_evaluation(configs)
    print("\nComparative Results:")
    print(comparison_df)