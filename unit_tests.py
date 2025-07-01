import unittest
import numpy as np
from PIL import Image
from nestplaatsing import (
    load_model,
    load_image,
    select_nest_locations,
    print_results
)

class TestNestPlaatsing(unittest.TestCase):

    def test_load_image(self):
        image = load_image(r"C:\Users\moham\Nest-Project\nest_data\test\images\test_3.jpg")  # Je moet dit pad aanpassen naar een bestaande testafbeelding
        self.assertIsInstance(image, Image.Image)

    def test_select_nest_locations(self):
        pixel_scale = 0.0175  #  meter per pixel schaal
        mask = np.zeros((100, 160), dtype=bool)
        mask[50] = True  # Eén rij als toegestaan gebied

        nests = select_nest_locations(mask, pixel_scale, min_distance_m=3.0, max_nests=5)

        # Test slaagt zolang er minstens 1 nest is gevonden
        self.assertGreaterEqual(len(nests), 1)

    def test_print_results(self):
        nests = [(10, 20), (40, 50)]
        species = "huismus"
        pixel_scale = 0.1
        # Test gewoon dat de functie zonder fout draait
        try:
            print_results(nests, species, pixel_scale)
        except Exception as e:
            self.fail(f"print_results() raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
