def mn(points, l, r, key):
    if(l==r):
        return points[l]
    mid = (l+r)//2
    p1  = mn(points, l, mid, key)
    p2 = mn(points, mid+1, r, key)
    if(key(p1)<key(p2)):
        return p1
    else:
        return p2

def mx(points, l, r, key):
    if(l==r):
        return points[l]
    mid = (l+r)//2
    p1  = mx(points, l, mid, key)
    p2 = mx(points, mid+1, r, key)
    if(key(p1)>key(p2)):
        return p1
    else:
        return p2

print(mn([[1,2], [2,3], [5,4],[-1,20]], 0, 3, lambda x: x[0]))