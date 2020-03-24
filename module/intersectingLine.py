import collections

Params = collections.namedtuple('Params', ['a', 'b', 'c']) #to store equation of a line
maxDistance = 10
sensitive = 1

def calcParams(point1, point2): #line's equation Params computation
    if point2[1] - point1[1] == 0:
         a = 0
         b = -1.0
    elif point2[0] - point1[0] == 0:
        a = -1.0
        b = 0
    else:
        a = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = -1.0

    c = (-a * point1[0]) - b * point1[1]
    return Params(a, b, c)

def areLinesIntersecting(params1, params2, point1, point2, linePoint):
    det = params1.a * params2.b - params2.a * params1.b
    if det == 0:
        return False #lines are parallel
    else:
        x = (params2.b * -params1.c - params1.b * -params2.c)/det
        y = (params1.a * -params2.c - params2.a * -params1.c)/det
        min_x = min(point1[0], point2[0]) - sensitive
        max_x = max(point1[0], point2[0]) + sensitive
        min_y = min(point1[1], point2[1]) - sensitive
        max_y = max(point1[1], point2[1]) + sensitive
        if (min_x < int(x) < max_x) and (min_y < int(y) < max_y):
            min_lx = min(linePoint[0][0], linePoint[1][0]) - maxDistance
            max_lx = max(linePoint[0][0], linePoint[1][0]) + maxDistance
            min_ly = min(linePoint[0][1], linePoint[1][1]) - maxDistance
            max_ly = max(linePoint[0][1], linePoint[1][1]) + maxDistance
            if (min_lx <= int(x) <= max_lx) and (min_ly <= int(y) <= max_ly):
                return True #lines are intersecting inside the line segment
            else:
                # print('intersected but not in range')
                # print('x : %s, y : %s, linePoint : %s' % (str(x), str(y), str(linePoint)))
                return False
        else:
            # print('no intersected')
            # print('x:%s, y:%s, min_x:%s, max_x:%s, min_y:%s, max_y:%s' % (str(int(x)), str(int(y)), str(int(min_x)), str(int(max_x)), str(int(min_y)), str(int(max_y))))
            return False #lines are intersecting but outside of the line segment
