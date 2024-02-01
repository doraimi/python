class ClassA:
    def __init__(self,paraA):
        print("__init__")
        self.innerA=paraA
    def outputTest(self , paraB):
        print("outputTest")
        print(f"output >> {self.innerA}")
        print(f"output >> {paraB}")
        print (f"ClassA __name__{__name__}")