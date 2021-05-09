
app8 = open( "acc_app_8.csv","r")
app10 = open("acc_app_10.csv","r")
app12 = open("acc_app_12.csv","r")
app14 = open("acc_app_14.csv","r")
app16 = open("acc_app_16.csv","r")

files = [app8, app10, app12, app14, app16]


for a,b,c,d,e in zip(app8, app10, app12, app14, app16):
    print(f"{a.strip()},{b.strip()},{c.strip()},{d.strip()},{e.strip()}")