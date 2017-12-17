import pygame

pygame.init()
# initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
myfont = pygame.font.SysFont("monospace", 40)


analyzing=False

pygame.display.init()
screenSize= (720, 576)
screen = pygame.display.set_mode( ( screenSize ) )
running=True


while running:

	# render text
	label = myfont.render("Some cóño text!", 1, (255,255,255))
	screen.blit(label, (100, 100))

	for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False  # Be interpreter friendly
			elif event.type == pygame.KEYDOWN:
				print("KEYDOWN!")
				if event.key == pygame.K_ESCAPE:
					running = False
				if event.key == pygame.K_p:
					if analyzing:
						analyzing=False
					else:
						analyzing=True
					print("analyzing:",analyzing)

	pygame.display.update()