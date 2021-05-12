from utils import *
from postprocessing import *
from eval import *
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


for fidx in range(10):
    create_template(10, fidx)
