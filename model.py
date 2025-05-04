import numpy as np
from scipy.stats.qmc import Sobol, PoissonDisk
from sklearn.metrics import pairwise_distances

class Geometry:
    """
    This class prepares a geometry
    - an arena that is a circle of radius 5
    centered at the origin
    and with 0,1,2 or 3 holes in it
    - circles of radius 1
    with centers at points 2.5 distance
    away from the origin, in directions
    of corresponding roots of unity (for 2,3 holes)
    or at the origin (for 1 hole)
    """
    
    def __init__(self, n_holes):
        if n_holes not in [0,1,2,3]:
            raise Exception('Can only do for n_holes from 0 to 3')
        else:
            self.n_holes = n_holes
            
    def indicator(self, points):
        xs, ys = points[:,0], points[:,1]
        """
        Indicator function checking
        whether a given point belongs 
        to the geometry
        
        Input: xs,ys - coordinates of the points
        Output: True / False - belongs to geometry or not
        """
        if self.n_holes == 0:
            return (xs**2 + ys**2) <= 5**2
            
        elif self.n_holes == 1:
            inside_big_circle = ((xs**2 + ys**2) <= 5**2)
            # one hole at the origin of radius 1"
            outside_hole = ((xs**2 + ys**2) > 1)
            # * = logical AND
            return inside_big_circle * outside_hole
        
        elif self.n_holes == 2:
            inside_big_circle = ((xs**2 + ys**2) <= 5**2)
            # first hole with center at (2.5, 0) of radius 1
            outside_right_hole = (((xs-2.5)**2 + ys**2) > 1)
            # second hole with center at (-2.5, 0) of radius 1
            outside_left_hole = (((xs+2.5)**2 + ys**2) > 1)
            # * = logical AND
            return inside_big_circle * outside_right_hole * outside_left_hole
        
        elif self.n_holes == 3:
            inside_big_circle = ((xs**2 + ys**2) <= 5**2)
            # zeroth hole with center at (2.5, 0) of radius 1
            outside_zeroth_hole = (((xs-2.5)**2 + ys**2) > 1)
            # first hole at center = 2.5*first_root_of_unity of radius 1
            xc1, yc1 = 2.5*np.cos(2*np.pi/3), 2.5*np.sin(2*np.pi/3)
            outside_first_hole = (((xs-xc1)**2 + (ys-yc1)**2) > 1)
            # second hole at center = 2.5*second_root_of_unity of radius 1
            xc2, yc2 = 2.5*np.cos(2*2*np.pi/3), 2.5*np.sin(2*2*np.pi/3)
            outside_second_hole = (((xs-xc2)**2 + (ys-yc2)**2) > 1)
            # * = logical AND
            return inside_big_circle * outside_zeroth_hole * outside_first_hole * outside_second_hole
            
                
    def sample_uniform(self, n_points=1024):
        n = n_points
        if not ((n & (n-1) == 0) and n != 0):
            raise Exception("Please choose n_points that is a power of 2")
        
        generator = Sobol(d=2)
        points = generator.random(n_points)
        
        # everything is in a square with side 10
        points *= 10
        # centered at the origin
        points -= 5
        
        # check if belongs to geometry
        valid_points = points[self.indicator(points)]
        
        return valid_points
    
    def sample_Pois(self, radius=0.2, n_points=1024):
        # TODO: generator with set seed
        rng = np.random.default_rng()
        sampler = PoissonDisk(d=2, radius=radius)
        
        points = sampler.random(n_points)
        
        # everything is in a square with side 10
        points *= 10
        # centered at the origin
        points -= 5
        
        # check if belongs to geometry
        valid_points = points[self.indicator(points)]
        
        return valid_points
    

class PlaceCellsModel:
    def __init__(self, n_holes, move_time_frac=0.8):
        self.geometry = Geometry(n_holes)
        self.move_time_frac = move_time_frac
        
        # TODO: add the arguments of this to the above method
        self.cells = self.geometry.sample_Pois(n_points = 256, radius = 0.05)
        self.n_cells = self.cells.shape[0]
        self.sigma_activation = 0.05
        self.sigma_noise = 0.01

    def sample_gaussian_walk(self, n_steps, sigma):
        if n_steps <= 0:
            raise Exception("Provide a positive integer number of steps")
        """
        steps are proposed from isotropic 2D Gaussian 
        accepted if remaining within the geometry,
        rejected otherwise
        """
        # starting at the origin
        #trajectory = np.array([[0.,0.]])
        
        # starting at random point, since origin is not
        # in the geometry if n_holes = 0
        trajectory = self.geometry.sample_uniform(n_points=8)[0:1,:]
        current_position = trajectory[-1,:]
        
        while n_steps > 0:
            # Bernoulli trial if to move or not
            if np.random.binomial(n=1, p=self.move_time_frac):
                step_proposal = sigma * np.random.normal(size=2)
                if self.geometry.indicator((current_position + step_proposal)[np.newaxis,:])[0]:
                    # also check that midpoint is within geometry
                    # so there are no teleportations through the holes
                    if self.geometry.indicator((current_position + step_proposal/2.)[np.newaxis,:])[0]:
                        new_position = current_position + step_proposal
                        trajectory = np.vstack((trajectory, new_position))
                        current_position = new_position
                        n_steps -= 1
            else:
                new_position = current_position
                trajectory = np.vstack((trajectory, new_position))
                current_position = new_position
                n_steps -= 1
        
        # origin is not in the geometry when n_holes=1
        return trajectory
    
    def sample_Levy_walk(self, n_steps, min_step=0.1, max_step=1., exponent=1.5):
        if n_steps <= 0:
            raise Exception("Provide a positive integer number of steps")
        """
        steps are cproposed from isotropic 2D Gaussian 
        accepted if remaining within the geometry,
        rejected otherwise
        """
        # starting at the origin
        #trajectory = np.array([[0.,0.]])
        
        # starting at random point, since origin is not
        # in the geometry if n_holes = 0
        trajectory = self.geometry.sample_uniform(n_points=8)[0:1,:]
        current_position = trajectory[-1,:]
        
        while n_steps > 0:
            # Bernoulli trial if to move or not
            if np.random.binomial(n=1, p=self.move_time_frac):
                direction = np.random.normal(size=2)
                direction /= np.linalg.norm(direction)
                step_size = min_step + (max_step-min_step)*(1-np.random.power(exponent+1,size=1))
                step_proposal = step_size * direction
                if self.geometry.indicator((current_position + step_proposal)[np.newaxis,:])[0]:
                    # also check that midpoint is within geometry
                    # so there are no teleportations through the holes
                    if self.geometry.indicator((current_position + step_proposal/2.)[np.newaxis,:])[0]:
                        new_position = current_position + step_proposal
                        trajectory = np.vstack((trajectory, new_position))
                        current_position = new_position
                        n_steps -= 1
            else:
                new_position = current_position
                trajectory = np.vstack((trajectory, new_position))
                current_position = new_position
                n_steps -= 1
         
        # origin is not in the geometry when n_holes=1
        return trajectory
    
    # TODO: make these into **kwargs
    def sample_signal(self, n_steps=300, min_step=0.1, max_step=3., exponent=1.5):
        # points of the trajectory
        traj = self.sample_Levy_walk(n_steps=n_steps, min_step=0.1, max_step=3., exponent=1.5)

        cells = self.cells 
        sigma_a = 1 #2*self.sigma_activation
        
        d = pairwise_distances(cells, traj)
        
        # Gaussian activation
        ps = np.exp(-d**2 / (2*(sigma_a**2)))/(np.sqrt(2*np.pi)*sigma_a)
        
        # rejection sampling trick
        u = np.random.uniform(size = (cells.shape[0], traj.shape[0]))
        s = (u <= ps).astype(float).T

        return s