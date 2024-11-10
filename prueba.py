import unittest
from pipeline import *

class TestEnsayos(unittest.TestCase):
    def test_pipe(self):
        history, mejor_modelo, mejor_acc, valores_ROC = final()
        train_accuracy = round(history.history['accuracy'][-1],2)
        test_accuracy = round(history.history['val_accuracy'][-1],2)
        valores_ROC.to_csv("ensayos_valores_roc.csv")
        self.assertLessEqual(np.abs(train_accuracy-test_accuracy), 10,
                                 print("No presenta Underfitting ni Overfitting"))
        print(f'El mejor modelo segun el accuracy global es el {mejor_modelo} con un accuracy del {mejor_acc}%, los valores ROC para cada tema y modelo estan en el archivo "ensayos_valores_roc.csv"')

if __name__ == "__main__":
    unittest.main()