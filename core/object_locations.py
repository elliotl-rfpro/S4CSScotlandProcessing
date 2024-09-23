"""File to store object locations within simulated/measured images.
First element is simulated, second element is measured."""

"""Winter Test Site, Camera 1"""
c1_road_target_near = [[883, 1084, 1214, 1512], [625, 1003, 1039, 1540]]
c1_road_target_far = [[1536, 936, 1626, 1053], [1412, 903, 1515, 1036]]
c1_bw_target = [[1218, 1230, 1389, 1402], [1041, 1173, 1271, 1397]]
c1_grass = [[1661, 1695, 2000, 1850], [1517, 1477, 1868, 1627]]
c1_wall = [[35, 1071, 235, 1185], [2007, 956, 2208, 1049]]
c1_trees = [[851, 413, 1327, 862], [276, 114, 828, 672]]
c1_all = [c1_road_target_near, c1_road_target_far, c1_bw_target, c1_grass, c1_wall, c1_trees]

"""Winter Test Site, Camera 2"""
c2_road_target_near = [[], []]
c2_road_target_far = [[], []]
c2_bw_target = [[], []]
c2_grass = [[], []]
c2_trees = [[], []]
c2_sky = [[], []]
c2_hills = [[], []]
c2_all = [c2_road_target_near, c2_road_target_far, c2_bw_target, c2_grass, c2_trees, c2_sky, c2_hills]

"""Winter Test Site, RAW Camera"""
c3_road_target_near = [[1743, 930, 1842, 1065], [1648, 964, 1771, 1122]]
c3_road_target_far = [[1189, 964, 1236, 1024], [1216, 992, 1258, 1043]]
c3_bw_target = [[1369, 992, 1404, 1028], [1371, 1007, 1404, 1040]]
c3_grass = [[1200, 1175, 1500, 1475], [1160, 1140, 1360, 1340]]
c3_trees = [[], []]
c3_sky = [[], []]
c3_hills = [[], []]
c3_all = [c3_road_target_near, c3_road_target_far, c3_bw_target, c3_grass, c3_trees, c3_sky, c3_hills]
