from faker import Faker
from faker.providers import BaseProvider
import random
import csv
import os


class HeadacheGenerator(BaseProvider):
    def isHeadache(self):
        return random.choice([True, False])


class SneezeGenerator(BaseProvider):
    def isSneeze(self):
        return random.choice([True, False])


class FeverGenerator(BaseProvider):
    def isFever(self):
        return random.choice([True, False])


class ColdFeetGenerator(BaseProvider):
    def isColdFeet(self):
        return random.choice([True, False])


class isFluGenerator(BaseProvider):
    def isFlu(self):
        return random.choice([True, False])


fake = Faker()

fake.add_provider(HeadacheGenerator)
fake.add_provider(SneezeGenerator)
fake.add_provider(FeverGenerator)
fake.add_provider(ColdFeetGenerator)
fake.add_provider(isFluGenerator)


def main():
    path_to_folder = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'testcase'))
    path_to_file = os.path.abspath(
        os.path.join(path_to_folder, "is_flu.csv"))
    with open(path_to_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["isHeadache", "isSneeze",
                        "isFever", "isColdFeet", "isFlu"])
        # Only create 5 combinations that results to flu, everything else is NOT Flu
        # Headache + Sneeze + Fever + ColdFeet = Flu
        # Headache + Sneeze + Fever + NOT ColdFeet = Flu
        # Headache + Sneeze + NOT Fever + ColdFeet = Flu
        # Headache + Sneeze + NOT Fever + NOT ColdFeet = Flu
        # NOT Headache + Sneeze + Fever + ColdFeet = Flu
        for i in range(50):
            hasHeadache = fake.isHeadache()
            hasSneeze = fake.isSneeze()
            hasFever = fake.isFever()
            hasColdFeet = fake.isColdFeet()
            hasFlu = fake.isFlu()
            if hasHeadache == hasSneeze == hasFever == hasColdFeet == True:
                hasFlu = True
            elif hasHeadache == hasSneeze == hasFever == True and hasColdFeet == False:
                hasFlu = True
            elif hasHeadache == hasSneeze == hasColdFeet == True and hasFever == False:
                hasFlu = True
            elif hasHeadache == hasSneeze == True and hasFever == hasColdFeet == False:
                hasFlu = True
            elif hasHeadache == False and hasSneeze == hasFever == hasColdFeet == True:
                hasFlu = True
            else:
                hasFlu = False
            writer.writerow(
                [hasHeadache, hasSneeze, hasFever, hasColdFeet, hasFlu])
        f.close()


main()
