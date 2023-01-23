class Option():

    @staticmethod
    def get_id(i: int) -> str:
        if i % 2 == 0:
            id = "ContentPlaceHolder1_ctl03_rptData2" + "_itemTR_" + str(i)
        else:
            id = "ContentPlaceHolder1_ctl03_rptData2" + "_altitemTR_" + str(i)
        return id