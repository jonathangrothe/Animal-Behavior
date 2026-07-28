import unittest
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from python_scripts.simulation_metrics import get_bifurcation_angle
from helper_functions.helpers import get_estimated_angle

def solve_curve_start(x0,y0, x1,y1, x2,y2, dec_len):
    # helper function for setup, solves for where the curve should start to cut corners
    
    a1 = np.atan2(y1-y0,x1-x0)
    a2 = np.atan2(y2-y1,x2-x1)
    aprox_angle = get_estimated_angle(a1,a2) 
    dec_factor = aprox_angle*0.2/np.pi 

    if np.abs(a2 - a1) > np.pi:
        if a2 <0 and a1 >0:
            a2 += 2*np.pi
        if a2 >0 and a1 < 0:
            a2 -= 2*np.pi
    p0 = np.array([x0,y0])
    p2 = np.array([x2,y2])
    n = dec_len
    if n-1 <1:
        d = a2-a1
    else:
        d = (a2 - a1) / (n - 1)
    scale = np.sin(n*d/2) / np.sin(d/2)
    mid_angle = a1 + (n-1)*d/2
    D = dec_factor * scale * np.array([np.cos(mid_angle),np.sin(mid_angle)])
    u1 = np.array([np.cos(a1), np.sin(a1)])
    u2 = np.array([np.cos(a2), np.sin(a2)])
    t, s = np.linalg.solve(np.column_stack([u1, -u2]), (p2 - p0) - D)
    start_points = p0 + t*u1
    finish_points = start_points + D

    return start_points[0], start_points[1], dec_factor, aprox_angle 
    
def coords_setup(xpts,ypts,line_len,dec_len):
    a1 = np.atan2(ypts[1]-ypts[0],xpts[1]-xpts[0])
    x0 = xpts[0]
    y0 = ypts[0]
    x_traj = []
    y_traj = []
    directions = [a1]
    n_bifs = len(xpts)-2
    point_array = np.zeros((7,n_bifs))
    for i in range(n_bifs):
        x1 = xpts[i+1]
        y1 = ypts[i+1]
        x2 = xpts[i+2]
        y2 = ypts[i+2]
        a2 = np.atan2(y2-y1,x2-x1)
        directions.append(a2)
        if np.abs(a2-a1) > np.pi:
            if a2 < a1:
                a2 += 2*np.pi
            else:
                a2 -= 2*np.pi
        if dec_len[i] > 1:
            x1_gr,y1_gr,dec_factor1, flat_angle = solve_curve_start(x0,y0,x1,y1,x2,y2,dec_len[i])
            point_array[0,i] = flat_angle
            point_array[1,i] = x0
            point_array[2,i] = y0
            point_array[3,i] = x1
            point_array[4,i] = y1
            if i > 0:
                point_array[5,i-1] = x1_gr
                point_array[6,i-1] = y1_gr
            x_gr_prebif = np.linspace(x0,x1_gr,num=line_len[i])
            y_gr_prebif = np.linspace(y0,y1_gr,num=line_len[i])
            x_traj.append(x_gr_prebif)
            y_traj.append(y_gr_prebif)
            bif_range = np.linspace(a1,a2,num=dec_len[i])
            x_gradualfirst = [x1_gr]
            y_gradualfirst = [y1_gr]
            for item in bif_range:
                x_gradualfirst.append(x_gradualfirst[-1]+dec_factor1*np.cos(item))
                y_gradualfirst.append(y_gradualfirst[-1]+dec_factor1*np.sin(item))
            x_traj.append(x_gradualfirst)
            y_traj.append(y_gradualfirst)
            x0 = x_gradualfirst[-1]
            y0 = y_gradualfirst[-1]
        else:
            _, _, _, flat_angle_im = solve_curve_start(x0,y0,x1,y1,x2,y2,dec_len[i])
            point_array[0,i] = flat_angle_im
            x_straight_through = np.linspace(x0,x1,num=line_len[i])
            y_straight_through = np.linspace(y0,y1,num=line_len[i])
            x_traj.append(x_straight_through)
            y_traj.append(y_straight_through)
            x0 = x1
            y0 = y1

        a1 = a2
    final_gradual_x = np.linspace(x0,xpts[-1],line_len[-1])
    final_gradual_y = np.linspace(y0,ypts[-1],line_len[-1])
    x_traj.append(final_gradual_x)
    y_traj.append(final_gradual_y)
    final_x_traj = np.concat(x_traj)
    final_y_traj = np.concat(y_traj)
    point_array[5,n_bifs-1] = xpts[-1]
    point_array[6,n_bifs-1] = ypts[-1]
    
    total_time = len(final_x_traj)
    x_diff = np.diff(final_x_traj)
    y_diff = np.diff(final_y_traj)
    headings = np.zeros(total_time)
    headings[0] = np.atan2(ypts[1]-ypts[0],xpts[1]-xpts[0])
    for index in range(total_time-1):
        if (np.abs(x_diff[index]) < 1e-9) or (np.abs(y_diff[index]) < 1e-9):
            headings[index+1] = headings[index]
        else:
            head_angle = np.atan2(y_diff[index],x_diff[index])
            if head_angle < 0: 
                head_angle += 2*np.pi
            headings[index+1] = head_angle
    headings = np.unwrap(headings)
    theor_corner1 = []
    theor_bif_angle = []
    theor_corner2 = []
    for p in range(len(xpts)-2):
        if dec_len[p] > 1:
            angle = point_array[0,p]
            p0_x = point_array[1,p]
            p0_y = point_array[2,p]
            p1_x = point_array[3,p]
            p1_y = point_array[4,p]
            p2_x = point_array[5,p]
            p2_y = point_array[6,p]
            hyp = np.sqrt((p2_x-p0_x)**2+(p2_y-p0_y)**2)
            d1 = np.sqrt((p1_x-p0_x)**2+(p1_y-p0_y)**2)
            d2 = np.sqrt((p2_x-p1_x)**2+(p2_y-p1_y)**2)
            start_corner = np.asin((d2*np.sin(angle))/hyp)
            end_corner = np.asin((d1*np.sin(angle))/hyp)
            theor_corner1.append(start_corner)
            theor_bif_angle.append(angle)
            theor_corner2.append(end_corner)
        else:
            theor_corner1.append(-1)
            theor_bif_angle.append(point_array[0,p])
            theor_corner2.append(-1)

    np.set_printoptions(precision=5)
    #print(f"coords setup start:  {theor_corner1}")
    #print(f"coords setup angles: {theor_bif_angle}")
    #print(f"coords setup end:    {theor_corner2}")
    return [final_x_traj,final_y_traj,headings], directions, [theor_bif_angle,theor_corner1,theor_corner2]

class ToleranceTesting(unittest.TestCase):
    '''
    testing some simple two bifurcation cases where the trajectory moves smoothly between three angles in a pattern of line, turn, line, turn, line
    testing to ensure that our cutoff for what counts as a bifurcation is consistent with how we think it should work
    the idea is that our get_bifurcation_angle function should detect the start and end points of the turn, 
    but if the start of this curve and the start of the next curve are directionally similar, then we shouldn't count this point as a bifurcation
    '''

    def test_allquadrants(self):
        for q in range(1):
            rng = np.random.default_rng()
            a1 = rng.uniform(0, np.pi/2)
            a2 = rng.uniform(0, np.pi/2)
            a3 = rng.uniform(0, np.pi/2)
            #a1 = 0.09391558286929175
            #a2 = 1.556727091428781
            #a3 = 0.1469366738858174
            print(f"a1: {a1}, a2: {a2}, a3: {a3}") 
            a1_list = []
            a2_list = []
            a3_list = []
            xpts_arr = np.zeros((64,4))
            ypts_arr = np.zeros((64,4))
            x0 = 50
            y0 = 50
            xpts_arr[:,0] = x0
            ypts_arr[:,0] = y0
            dist = 10
            line_lens = [30,30,30]
            dec_lens = [30,30]
            for i in range(4):
                a1_list.append(a1+(np.pi/2)*i)
                a2_list.append(a2+(np.pi/2)*i)
                a3_list.append(a3+(np.pi/2)*i)
            for ind1,item1 in enumerate(a1_list):
                for ind2,item2 in enumerate(a2_list): 
                    for ind3,item3 in enumerate(a3_list):
                        x1 = x0 + dist*np.cos(item1)
                        y1 = y0 + dist*np.sin(item1)
                        x2 = x1 + dist*np.cos(item2)
                        y2 = y1 + dist*np.sin(item2)
                        x3 = x2 + dist*np.cos(item3)
                        y3 = y2 + dist*np.sin(item3)
                        row = 16*ind1+4*ind2+ind3
                        xpts_arr[row,1] = x1
                        xpts_arr[row,2] = x2
                        xpts_arr[row,3] = x3
                        ypts_arr[row,1] = y1
                        ypts_arr[row,2] = y2
                        ypts_arr[row,3] = y3
            direction_thresh = [np.pi/1000,np.pi/20,np.pi/10,np.pi/2,np.pi]
            for thresh in direction_thresh:
                for r in range(64):
                    x = xpts_arr[r,:]
                    y = ypts_arr[r,:]
                    coords, directions, theor_angle = coords_setup(x,y,line_lens,dec_lens)
                    ind, angle = get_bifurcation_angle(coords[0],coords[1],coords[2],thresh)
                    bif_angles = theor_angle[0]
                    directions_unwrapped = np.unwrap(directions)
                    b1_turn_valid = True
                    b2_turn_valid = True
                    diff1 = np.abs(directions_unwrapped[1]-directions_unwrapped[0])
                    new_d1 = directions_unwrapped[0]
                    new_d2 = directions_unwrapped[1]
                    if (diff1/(dec_lens[0]-1) < np.pi/720):
                        b1_turn_valid = False 
                        adjustment = (1 if directions_unwrapped[1]-directions_unwrapped[0] > 0 else -1)*theor_angle[1][0]
                        new_d1 = new_d1 + adjustment
                    diff2 = np.abs(directions_unwrapped[2]-directions_unwrapped[1])
                    if (diff2/(dec_lens[1]-1) < np.pi/720):
                        b2_turn_valid = False
                        adjustment = (1 if directions_unwrapped[2]-directions_unwrapped[1] > 0 else -1)*theor_angle[1][1]
                        new_d2 = new_d2 + adjustment
                    if not b1_turn_valid and b2_turn_valid:
                        directions_unwrapped = np.unwrap([new_d1,directions_unwrapped[2]])
                    if b1_turn_valid and not b2_turn_valid:
                        directions_unwrapped = np.unwrap([directions_unwrapped[0],new_d2])
                    if not b1_turn_valid and not b2_turn_valid:
                        directions_unwrapped = [0]
                    b1_delta_valid = True
                    b2_delta_valid = True
                    if b1_turn_valid and b2_turn_valid:
                        diff1 = np.abs(directions_unwrapped[1]-directions_unwrapped[0])
                        if diff1 < thresh:
                            b1_delta_valid = False
                            adjustment = adjustment = (1 if directions_unwrapped[1]-directions_unwrapped[0] > 0 else -1)*theor_angle[1][0]
                            new_d1 = directions_unwrapped[0] + adjustment
                            directions_unwrapped = np.unwrap([new_d1,directions_unwrapped[2]])
                        diff2 = np.abs(directions_unwrapped[-1]-directions_unwrapped[-2])
                        if diff2 < thresh:
                            b2_delta_valid = False
                            if b1_delta_valid and b1_turn_valid:
                                adjustment = adjustment = (1 if directions_unwrapped[-1]-directions_unwrapped[-2] > 0 else -1)*theor_angle[1][1]
                                new_d2 = directions_unwrapped[1] + adjustment
                                directions_unwrapped = np.unwrap([directions_unwrapped[0],new_d2])
                    elif b1_turn_valid or b2_turn_valid:
                        diff = np.abs(directions_unwrapped[1]-directions_unwrapped[0])
                        if diff < thresh:
                            if not b1_turn_valid:
                                b2_delta_valid = False
                            else:
                                b1_delta_valid = False
                    b1_valid = b1_delta_valid and b1_turn_valid
                    b2_valid = b2_delta_valid and b2_turn_valid
                    if b1_valid and b2_valid:
                        self.assertEqual(2,len(ind))
                        self.assertEqual(2,len(angle))
                        self.assertAlmostEqual(bif_angles[0],angle[0])
                        self.assertAlmostEqual(bif_angles[1],angle[1])
                    if b1_valid and not b2_valid: 
                        self.assertEqual(1,len(ind))
                        self.assertEqual(1,len(angle))
                        print(f"first valid, second not valid: {angle}, theor angles: {theor_angle[0]}")
                        within_tol1 = np.abs(bif_angles[0]-angle[0]) <= diff2
                        self.assertEqual(True,within_tol1)
                    if not b1_valid and b2_valid:
                        if len(ind) != 1:
                            plt.figure(1)
                            plt.plot(x,y)
                            plt.show()
                        self.assertEqual(1,len(ind))
                        self.assertEqual(1,len(angle))
                        print(f"first not valid, second valid: {angle}, theor angles: {theor_angle[0]}")
                        within_tol2 = np.abs(bif_angles[1]-angle[0]) <= diff1
                        self.assertEqual(True,within_tol2)
                    if not b1_valid and not b2_valid:
                        if len(ind) != 0:
                            plt.figure(1)
                            plt.plot(x,y)
                            plt.show()
                        self.assertEqual(0,len(ind))
                        self.assertEqual(0,len(angle))

class Dimensions(unittest.TestCase):
    '''
    dimensions tests
    '''
    # TO do: add testing for headings
    def test_dimensions(self): 
        '''
        Testing to ensure the dimensions of xPos and yPos line up
        '''
        print("Dimensions")
        xPos = np.array([50]*10)
        yPos = np.array([50]*8)
        yPos_correct = np.linspace(50,80,num=10)
        headings = np.array([np.pi/2]*10)
        with self.assertRaises(ValueError):
            indices, bif_angle = get_bifurcation_angle(xPos,yPos,headings)
        correct_ind, correct_angle = get_bifurcation_angle(xPos,yPos_correct,headings)
        self.assertEqual(0,len(correct_ind))
        self.assertEqual(0,len(correct_angle))

class Thresholds(unittest.TestCase):
    '''
    Testing the threshold for what proportion of ddirection is needed to be considered above the threshold
    '''

    def test_boundary_single(self):
        '''
        Testing the boundaries for a bifurcation being above or below the threshold for a simple case of single bifurcation
        '''
        xpts_list = [50,50,40]
        ypts_list = [0,50,60]
        line_lens = [50,30]
        dec_lensimm = np.array([1])
        dec_lensgr = np.array([10])
        imm_coords, directions_im, flat_angles_im = coords_setup(xpts_list,ypts_list,line_lens,dec_lensimm)
        gradual_coords, directions, flat_angles = coords_setup(xpts_list,ypts_list,line_lens,dec_lensgr)
        self.assertAlmostEqual(flat_angles_im[0][0], flat_angles[0][0])
        self.assertAlmostEqual(directions_im[0],directions[0])
        self.assertAlmostEqual(directions_im[1],directions[1])
        direction_diff = np.abs(directions[1]-directions[0])
        thresholds = np.zeros(13)
        thresholds[0:12] = np.linspace(0.01,np.pi+0.1,num=12)
        thresholds[12] = direction_diff
        for item in thresholds:
            ind_im, angle_im = get_bifurcation_angle(imm_coords[0],imm_coords[1],imm_coords[2],item)
            ind_gr, angle_gr = get_bifurcation_angle(gradual_coords[0],gradual_coords[1],gradual_coords[2],item)
            if direction_diff >= item:
                self.assertEqual(1,len(ind_im))
                self.assertEqual(1,len(ind_gr))
                self.assertEqual(1,len(angle_im))
                self.assertEqual(1,len(angle_gr))
                self.assertAlmostEqual(flat_angles[0][0],angle_im[0])
                self.assertAlmostEqual(flat_angles[0][0],angle_gr[0])
            else:
                self.assertEqual(0,len(ind_im))
                self.assertAlmostEqual(0,len(ind_gr))

    
    def test_boundary_double(self):
        xpts_list = [50,80,24,34]
        ypts_list = [50,90,30,65]
        line_lens = [20,20,20]
        dec_lens = np.array([10,10])
        dec_lens_im = np.array([1,1])
        im_total = sum(line_lens)-1 
        imm_coords, directions_im, flat_angles_im = coords_setup(xpts_list,ypts_list,line_lens,dec_lens_im)
        gradual_coords, directions, flat_angles = coords_setup(xpts_list,ypts_list,line_lens,dec_lens)
        self.assertAlmostEqual(flat_angles_im[0][0],flat_angles[0][0])
        self.assertAlmostEqual(flat_angles_im[0][1],flat_angles[0][1])
        self.assertAlmostEqual(directions_im[0],directions[0])
        self.assertAlmostEqual(directions_im[1],directions[1])
        self.assertAlmostEqual(directions_im[2],directions[2])
        b1_unwrapped = np.unwrap(directions[0:2])
        b1_diff = np.abs(b1_unwrapped[1]-b1_unwrapped[0])
        b2_unwrapped = np.unwrap(directions[1:3])
        b2_diff = np.abs(b2_unwrapped[1]-b2_unwrapped[0])
        thresholds = np.zeros(30)
        thresholds[0:28] = np.linspace(0.001,np.pi+0.01, num=28)
        thresholds[28] = np.abs(directions[1]-directions[0])
        thresholds[29] = np.abs(directions[2]-directions[1])
    
        for item in thresholds: 
            ind_imm, angles_imm = get_bifurcation_angle(imm_coords[0],imm_coords[1],imm_coords[2],item)
            ind_gradual, angles_gradual = get_bifurcation_angle(gradual_coords[0],gradual_coords[1],gradual_coords[2],item)
            b1_valid = b1_diff >= item
            b2_valid = b2_diff >= item
            if b1_valid and b2_valid: 
                self.assertEqual(2,len(ind_imm))
                self.assertEqual(2,len(ind_gradual))
                self.assertEqual(2,len(angles_imm))
                self.assertEqual(2,len(angles_gradual))
                self.assertAlmostEqual(flat_angles[0][0],angles_imm[0])
                self.assertAlmostEqual(flat_angles[0][0],angles_gradual[0])
                self.assertAlmostEqual(flat_angles[0][1],angles_imm[1])
                self.assertAlmostEqual(flat_angles[0][1],angles_gradual[1])
            if b1_valid and not b2_valid:
                self.assertEqual(1,len(ind_imm))
                self.assertEqual(1,len(ind_gradual))
                self.assertEqual(1,len(angles_imm))
                self.assertEqual(1,len(angles_gradual))
                self.assertAlmostEqual(flat_angles[0][0],angles_imm[0])
                self.assertAlmostEqual(flat_angles[0][0],angles_gradual[0])
            if not b1_valid and b2_valid:
                self.assertEqual(1,len(ind_imm))
                self.assertEqual(1,len(ind_gradual))
                self.assertEqual(1,len(angles_imm))
                self.assertEqual(1,len(angles_gradual))
                self.assertAlmostEqual(flat_angles[0][1],angles_imm[0])
                self.assertAlmostEqual(flat_angles[0][1],angles_gradual[0])
            if not b1_valid and not b2_valid:
                self.assertEqual(0,len(ind_imm))
                self.assertEqual(0,len(ind_gradual))
                self.assertEqual(0,len(angles_imm))
                self.assertEqual(0,len(angles_gradual))


class MovementThresholds(unittest.TestCase):
    '''
    Testing the threshold for beginning movement
    '''
    def test_late_movement(self):
        '''
        Testing cases when the movement doesn't start until late, if its still turning at the end then we won't count that as a bifurcation
        '''
        static_coords, static_directions, static_angles = coords_setup([50,50.99,80],[50,50,80],np.array([100,2]),np.array([1]))
        late_coords, late_directions, late_angles = coords_setup([50,50.99,80],[50,50,80],np.array([100,3]),np.array([1]))
        eleven_jump_coords, eleven_jump_directions, eleven_jump_angles = coords_setup([50,50.99,80,80],[50,50,80,85],np.array([100,12,2]),np.array([1,1]))
        print(f"static angles: {static_angles}, late_angles: {late_angles}, eleven_jump_angles: {eleven_jump_angles}")

        self.assertAlmostEqual(static_directions[0],late_directions[0])
        self.assertAlmostEqual(static_directions[0],eleven_jump_directions[0])
        self.assertAlmostEqual(static_directions[1],late_directions[1])
        self.assertAlmostEqual(static_directions[1],eleven_jump_directions[1])
        self.assertAlmostEqual(static_angles[0][0],late_angles[0][0])
        self.assertAlmostEqual(static_angles[0][0],eleven_jump_angles[0][0])

        ind_static, angles_static = get_bifurcation_angle(static_coords[0],static_coords[1],static_coords[2])
        ind_late, angles_late = get_bifurcation_angle(late_coords[0],late_coords[1],late_coords[2])
        ind_eleven_jump, angles_eleven_jump = get_bifurcation_angle(eleven_jump_coords[0],eleven_jump_coords[1],eleven_jump_coords[2])

        print(f"ind static: {ind_static}, angles_static: {angles_static}")
        print(f"ind late: {ind_late}, angles_late: {angles_late}")
        print(f"ind 11: {ind_eleven_jump}, angles 11: {angles_eleven_jump}")

        self.assertEqual(0,len(ind_static))
        self.assertEqual(0,len(angles_static))

        self.assertEqual(1,len(ind_late))
        self.assertEqual(1,len(angles_late))
        self.assertEqual(100,ind_late[0])
        self.assertAlmostEqual(late_angles[0],angles_late[0])

        self.assertEqual(1,len(ind_eleven_jump))
        self.assertEqual(1,len(angles_eleven_jump))
        self.assertEqual(100,ind_eleven_jump[0])
        self.assertAlmostEqual(eleven_jump_angles[0],angles_eleven_jump[0])
      
        

class MaxTime(unittest.TestCase): #NEEDS UPDATES
    '''
    Tests for the maxtime parameter
    '''
    def test_maxtime(self): 
        '''
        Tests for the maxtime parameter
        '''
        timeout_coords, timeout_angles, timeout_angles = coords_setup([50,70],[50,70],np.array([5001]),np.array([1]))
        reach_coords, reach_tol, reach_angles = coords_setup([50,70,80],[50,70,60],np.array([3000,2000]),np.array([1]))

        ind_max, angles_max = get_bifurcation_angle(timeout_coords[0],timeout_coords[1],timeout_coords[2])
        ind_reaches, angles_reaches = get_bifurcation_angle(reach_coords[0],reach_coords[1],reach_coords[2])
        ind_timecrunch, angles_timecrunch = get_bifurcation_angle(reach_coords[0],reach_coords[1],reach_coords[2])

        self.assertEqual(0,len(ind_max))
        self.assertEqual(0,len(angles_max))
        self.assertEqual(1,len(ind_reaches))
        self.assertEqual(1,len(angles_reaches))
        self.assertEqual(0,len(ind_timecrunch))
        self.assertEqual(0,len(angles_timecrunch))
    

class UnexpectedMovement(unittest.TestCase):
    '''
    Testing unexpected movement
    '''
    #def test_spiral(self):

    #def test_past(self):

    #def test_jagged(self):

    #def test_smooth(self):
