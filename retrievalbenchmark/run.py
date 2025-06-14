"""
Script to run IMPROVED RAGBench retrieval evaluation
"""

import logging
from pathlib import Path

# Import the IMPROVED evaluation framework
from retrievalbenchmark.eval import run_realistic_retrieval_evaluation

# Import your retriever
from pipelines.retrieval import Retriever

def main():
    """
    Main function to run the complete IMPROVED RAGBench evaluation
    """

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("🎯 Starting IMPROVED RAGBench Retrieval Evaluation")

    try:
        # 1. Initialize your retriever
        logger.info("Initializing retriever...")
        retriever = Retriever(model="openai")

        # 2. Configure CHALLENGING evaluation parameters
        datasets_to_test = ['covidqa', 'finqa']  # Multi-domain challenge
        samples_per_dataset = 15  # 30 total questions for testing

        logger.info(f"CHALLENGING Evaluation configuration:")
        logger.info(f"  Datasets: {datasets_to_test} (multi-domain)")
        logger.info(f"  Samples per dataset: {samples_per_dataset}")
        logger.info(f"  Total questions: {len(datasets_to_test) * samples_per_dataset}")
        logger.info("  🎯 Using 60% relevance threshold (realistic)")

        # 3. Run the IMPROVED evaluation
        logger.info("Running IMPROVED retrieval evaluation...")
        overall_df, dataset_df, all_results = run_realistic_retrieval_evaluation(
            retriever=retriever,
            datasets_to_test=datasets_to_test,
            samples_per_dataset=samples_per_dataset,
            chunk_type="token",
            chunk_size=400,
            chunk_overlap=0
        )

        # 4. Display key results
        print("\n" + "="*80)
        print("🎯 IMPROVED EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)

        if not overall_df.empty:
            print("\nQUICK SUMMARY:")
            for _, row in overall_df.iterrows():
                print(f"{row['retriever']:20} | MRR: {row.get('mrr', 0):.3f} | P@5: {row.get('precision@5', 0):.3f} | Hit Rate: {row.get('hit_rate', 0):.3f}")

            # Find best overall performer
            if 'mrr' in overall_df.columns:
                best_mrr_idx = overall_df['mrr'].idxmax()
                best_retriever = overall_df.loc[best_mrr_idx, 'retriever']
                best_mrr = overall_df.loc[best_mrr_idx, 'mrr']
                print(f"\n🏆 BEST OVERALL (MRR): {best_retriever} ({best_mrr:.4f})")

            if 'hit_rate' in overall_df.columns:
                best_hit_idx = overall_df['hit_rate'].idxmax()
                best_hit_retriever = overall_df.loc[best_hit_idx, 'retriever']
                best_hit_rate = overall_df.loc[best_hit_idx, 'hit_rate']
                print(f"🎯 BEST HIT RATE: {best_hit_retriever} ({best_hit_rate:.4f})")

            # Show realism check
            avg_mrr = overall_df['mrr'].mean()
            avg_hit_rate = overall_df['hit_rate'].mean()
            print(f"\n📊 REALISM CHECK:")
            print(f"Average MRR: {avg_mrr:.3f} {'✅ Realistic' if 0.15 <= avg_mrr <= 0.8 else '❌ Still unrealistic'}")
            print(f"Average Hit Rate: {avg_hit_rate:.3f} {'✅ Realistic' if 0.4 <= avg_hit_rate <= 0.95 else '❌ Still unrealistic'}")

        # 5. Show where results are saved
        results_dir = Path("./realistic_retrieval_evaluation")
        print(f"\n📁 Detailed results saved to: {results_dir.absolute()}")
        print("   - realistic_retrieval_summary.csv (overall performance)")
        print("   - realistic_dataset_breakdown.csv (per-dataset results)")
        print("   - realistic_detailed_results.json (full results)")

        logger.info("🎯 IMPROVED evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have the required imports:")
        print("   - ChunkEmbedder from emb.chunks")
        print("   - PopulateQueries from postgres.populate")
        print("   - ChunkingService from chunkers.chunker")
        print("2. Check your database connection")
        print("3. Verify your retriever is working properly")
        raise


def run_quick_test():
    """
    Run a very quick test with minimal data to verify everything works
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("🧪 Running quick IMPROVED test...")

    try:
        retriever = Retriever(model="openai")

        # Very small test - just 4 questions total, 2 domains
        overall_df, dataset_df, all_results = run_realistic_retrieval_evaluation(
            retriever=retriever,
            datasets_to_test=['covidqa', 'finqa'],  # 2 different domains
            samples_per_dataset=2,  # Just 2 questions per domain = 4 total
            chunk_type="token",
            chunk_size=400,
            chunk_overlap=0
        )

        print("✅ Quick IMPROVED test completed successfully!")
        print(f"Evaluated {len(all_results)} retrieval strategies")

        # Show quick results
        if not overall_df.empty:
            print("\nQuick Results:")
            for _, row in overall_df.iterrows():
                print(f"  {row['retriever']:15}: MRR={row.get('mrr', 0):.3f}")

        return True

    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        return False


def run_small_evaluation():
    """Run small but realistic evaluation"""
    logging.basicConfig(level=logging.INFO)
    retriever = Retriever(model="openai")

    print("\n📊 Running SMALL REALISTIC evaluation...")
    print("🎯 Multi-domain challenge: medical + financial")

    overall_df, dataset_df, all_results = run_realistic_retrieval_evaluation(
        retriever=retriever,
        datasets_to_test=['covidqa', 'finqa'],  # 2 domains
        samples_per_dataset=15,  # 30 total questions
    )
    print("✅ Small realistic evaluation completed!")


def run_full_evaluation():
    """Run full challenging evaluation"""
    logging.basicConfig(level=logging.INFO)
    retriever = Retriever(model="openai")

    print("\n🚀 Running FULL CHALLENGING evaluation...")
    print("🎯 Multi-domain challenge: medical + legal + financial + technical")

    overall_df, dataset_df, all_results = run_realistic_retrieval_evaluation(
        retriever=retriever,
        datasets_to_test=['covidqa', 'cuad', 'finqa', 'techqa'],  # 4 challenging domains
        samples_per_dataset=25,  # 100 total questions
    )
    print("✅ Full challenging evaluation completed!")


def run_mega_evaluation():
    """Run comprehensive evaluation across all domains"""
    logging.basicConfig(level=logging.INFO)
    retriever = Retriever(model="openai")

    print("\n🔥 Running MEGA COMPREHENSIVE evaluation...")
    print("🎯 ALL domains: medical + legal + financial + technical + general")

    overall_df, dataset_df, all_results = run_realistic_retrieval_evaluation(
        retriever=retriever,
        datasets_to_test=['covidqa', 'cuad', 'finqa', 'techqa', 'msmarco'],  # 5 domains
        samples_per_dataset=30,  # 150 total questions
    )
    print("✅ Mega comprehensive evaluation completed!")


if __name__ == "__main__":
    print("🎯 IMPROVED RAGBench Retrieval Evaluation")
    print("="*60)
    print("NEW FEATURES:")
    print("✅ 60% relevance threshold (realistic)")
    print("✅ Multi-domain testing")
    print("✅ Better similarity metrics")
    print("✅ Expected MRR: 0.2-0.7, Hit Rate: 0.5-0.9")
    print("="*60)

    # Ask user which mode to run
    mode = input("""Choose evaluation mode:
1. Quick test (4 questions, 2 domains)
2. Small realistic (30 questions, 2 domains) 
3. Full challenging (100 questions, 4 domains)
4. Mega comprehensive (150 questions, 5 domains)
5. Custom evaluation

Enter choice (1/2/3/4/5): """).strip()

    if mode == "1":
        print("\n🧪 Running quick test...")
        run_quick_test()

    elif mode == "2":
        print("\n📊 Running small realistic evaluation...")
        run_small_evaluation()

    elif mode == "3":
        print("\n🚀 Running full challenging evaluation...")
        run_full_evaluation()

    elif mode == "4":
        print("\n🔥 Running mega comprehensive evaluation...")
        run_mega_evaluation()

    elif mode == "5":
        print("\n⚙️ Running custom evaluation...")
        main()

    else:
        print("\nRunning default evaluation...")
        main()