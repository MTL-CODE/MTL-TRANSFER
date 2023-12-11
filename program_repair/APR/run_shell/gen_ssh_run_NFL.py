import os



file_path = "/home/LAB/yuhw/SE/TBar-master/SuspiciousCodePositions_new/"
type_names = []
for curDir, dirs, files in os.walk(file_path):
    type_name = curDir.split('/')[-1]
    type_names.append(type_name)
type_names = sorted(type_names)

for name_1 in type_names:
    print("java -Xmx1g -cp \"./*\" edu.lu.uni.serval.tbar.main.Main_NFL /home/LAB/yuhw/SE/TBar-master/D4J/projects_395/  "+ name_1 +"    /home/LAB/yuhw/SE/TBar-master/D4J/defects4j/    /home/LAB/yuhw/SE/TBar-master/SuspiciousCodePositions_526/    /home/LAB/yuhw/SE/TBar-master/FailedTestCases/    ")

for name_1 in type_names:
    print(name_1)



for name_2 in type_names[82:394]:
    print("timeout 3h java -Xmx1g -cp \"./*\" edu.lu.uni.serval.tbar.main.Main /home/LAB/yuhw/SE/TBar-master/D4J/projects_395/  "+ name_2 +"    /home/LAB/yuhw/SE/TBar-master/D4J/defects4j/         && ls   ")
