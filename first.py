import pygame
import time, math
import numpy as np
from pid import PID
import matplotlib.pyplot as plt



def onebyf_noise(samples=1024, scale=1):
	if samples < 2:
		return [0 for _ in range(samples)]
	def sub(s):
		f = [0 if f==0 else scale/f for f in range(samples)]
		f = np.array(f, dtype='complex')
		Np = (len(f) - 1) // 2
		phases = np.random.rand(Np) * 2 * np.pi
		phases_vec = np.cos(phases) + 1j * np.sin(phases)
		f[1:Np+1] *= phases_vec
		f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
		return np.fft.ifft(f).real
	
	def rms(arr):
		return math.sqrt(sum(x**2 for x in arr))

	return sub(samples)*scale * (1/rms(sub(samples)))


noise = onebyf_noise(1000, 1)
print(noise)
#plt.plot(noise)
#plt.show()
def rms(arr):
	return math.sqrt(sum(x**2 for x in arr))


rmss = [rms(onebyf_noise(samples)) for samples in range(1, 100)]
test = [1/math.pow(samples, 1/1.5 ) for samples in range(1, 100)]
#print(rmss)
plt.plot(rmss)
plt.plot(test)
plt.show()


for samples in range(100, 1000, 100):
	rms1 = rms(onebyf_noise(samples))
	rms2 = rms(onebyf_noise(samples, 1/rms1))
	print(samples, rms2)

exit()



screen = pygame.display.set_mode((1024, 1024))
screen.fill((255,255,255))
pygame.draw.line(screen, pygame.Color(255, 0, 0, 255), (100, 100), (500, 500), 50)
pygame.display.flip()
clock = pygame.time.Clock()

def world_to_pix(pos2):
	scale = 1
	screenx = 1024
	screeny=1024
	x, y = pos2
	return (int(x*scale)+screenx/2, screeny-(int(y*scale)+screeny/10))

class Robot(pygame.sprite.Sprite):
	WHEELBASE = 100 # mm
	WHEEL_RMS_ERROR = 50 # mm/s

	def add_wheel_error(self, w):
		err = min((math.fabs(2*w), self.WHEEL_RMS_ERROR)) # todo, this is confusing
		return w + np.random.normal(0, err)


	def __init__(self, position):
		pygame.sprite.Sprite.__init__(self)
		self.pos = position
		self.wheels_real = [0, 0]
		self.wheels_desired = self.wheels_real
		self.src_image = pygame.image.load('robot.png')
		self.src_image = pygame.transform.scale(self.src_image, (100, 100))
		self.pids = [PID(.1, 1, 0) for _ in self.wheels_real]


	def update(self, dt):
		# figure out what PID wants to send the wheels
		current = self.read_encoders()

		#desired = [pid.update(c, dt) for pid, c in zip(self.pids, current)]
		desired = self.wheels_desired # no pid

		#print(i, desired)
		self.wheels_real = [self.add_wheel_error(w) for w in desired]
		wheels = self.wheels_real

		forward = (wheels[0] + wheels[1])/2.0 * dt
		spin = (wheels[0] - wheels[1])*dt / self.WHEELBASE
		x, y, t = self.pos
		x += forward*math.sin(t)
		y += forward*math.cos(t)
		t += spin
		self.pos = (x, y, t)
		self.image = pygame.transform.rotate(self.src_image, -180/math.pi*self.pos[2])
		self.rect = self.image.get_rect()
		self.rect.center = world_to_pix(self.pos[0:2])

	def set_wheels(self, arr):
		self.wheels_desired = arr
		print(i, 'A')
		for pid2,w2 in zip(self.pids, arr):
			print(pid2)
			print(w2)
			pid2.setPoint(w2) 

	def read_encoders(self):
		return self.wheels_real # TODO no noise for now



robot = Robot((0, 0, 0))
robot_group = pygame.sprite.RenderPlain(robot)

timescale = 1
dt=1.0/30
for i in range(300):
	print(i)
	if i==2:
		robot.set_wheels([100, 100])
	if i==100:
		pass#robot.set_wheels([100, 0])


	_ = clock.tick(1/dt*timescale)
	screen.fill((255, 255, 255))
	robot_group.update(dt)
	robot_group.draw(screen)
	pygame.display.flip()
