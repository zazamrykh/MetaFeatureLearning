from src.mtft import MetaFeatExtractor, MetaFeatures
from src.data.dataset import Dataset
from src.utils.logger import get_logger
logger = get_logger(log_file='logs/exps/invariance_test.log')


extractor = MetaFeatExtractor()

dataset = Dataset.load('data/datasets/26.csv', 'data/datasets/26.meta.json', 26)
logger.info(f'Initial dataset columns: {dataset.df.columns}')
logger.info(f'Initial dataset first three rows: {dataset.df[:3]}')

mtft_before = extractor.extract(dataset)

shuffled_dataset = Dataset.load('data/datasets/26.csv', 'data/datasets/26.meta.json', 26)
shuffled_dataset.shuffle_dataset()
logger.info(f'Shuffled dataset columns: {shuffled_dataset.df.columns}')
logger.info(f'Shuffled dataset first three rows: {shuffled_dataset.df[:3]}')

mtft_shuffled = extractor.extract(shuffled_dataset)

base_result, base_dict = MetaFeatures.compare_metafeat(mtft_before.base_feat, mtft_shuffled.base_feat)
logger.info(f'Comparison base metafeatures result: {base_result}; problems: {base_dict}')

stat_result, stat_dict = MetaFeatures.compare_metafeat(mtft_before.stat_feat, mtft_shuffled.stat_feat)
logger.info(f'Comparison stat metafeatures result: {stat_result}; problems: {stat_dict}')

struct_result, struct_dict = MetaFeatures.compare_metafeat(mtft_before.struct_feat, mtft_shuffled.struct_feat)
logger.info(f'Comparison struct metafeatures result: {struct_result}; problems: {struct_dict}')

logger.info(f'Initial dataset features: {str(mtft_before)}')
logger.info(f'Shuffled dataset features: {str(mtft_shuffled)}')
