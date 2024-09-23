class regression_controller():
    def __init__(self, data):
        self.data = data

    def initalize_split(self, cv_type:str = None, test_size:float = None):
        from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
        while True:
            if cv_type == 'None':
                self.split = train_test_split
                if test_size == None:
                    test_size = float(input("Please enter the test size as a decimal: "))
                
                break
            elif cv_type == 'LeaveOneOut':
                self.split = LeaveOneOut
                break
            elif cv_type == 'KFold':
                self.split = KFold
                if test_size == None:
                    test_size = float(input("Please enter the size of splits as a decimal: "))
                    splits = int(1/test_size)
                break
            else:
                cv_type = input("Please enter a valid cross-validation type: LeaveOneOut, or KFold, or None: ")
                continue
        