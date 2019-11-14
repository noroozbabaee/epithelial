In this work, the aim is to create a Python version of the epithelial model (proximal convoluted tubule).The current epithelial model involves determination of 30 unknown, which includes solute concentrations, electrical potential and hydrostatic pressure.
In addition to the solute concentrations for the cell and interspace, we add more variables to the system to be more consistant to the main model. IMP,CBUF, HCBUF, VM added to the system variables(VE, PE and VI already exict).
The system is time independent, we just consider steady state conditions.
The aim is to solve the epithelial system through  Newton Method, applying Gaussian elimination and partial defferential equations.
All the differential equatione are presented in ERR_PHI_12112019_nok.py file.
ADD ERR_Pat_DEr_14112019_nok.py, we calculate the partial derivattive for all the differential equations presented in  ERR_PHI_12112019_nok.py file.
