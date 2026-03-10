def test_alpha():
    dform = [-2.0, -1.5, -1.0]
    sform = [0.5, 0.3, 0.1]
    smass = [500, 300, 200]
    freq = [0.5, 1.2, 2.0]
    area = 10.75
    cyy = 0.12
    a = 3

    freq = [6.28*w for w in freq]
    a = a/57.3
    cyy = cyy*57.3

    integral = 0
    for i in range(3):
        integral += dform[i]*sform[i]/(smass[i]*freq[i]**2)
    return - integral * area * a * cyy

print(test_alpha(), "рад")
print(test_alpha()*57.3, "град")
print(abs((test_alpha()*57.3)/3*100), "%")