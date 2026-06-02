import pygame

def draw_dashboard(screen, clock, pop: int, total_steps: int, mean_nodes: float, params: dict, width: int, dashboard_size: int, font, simulated_time_ms: int):
    """
    Displays the top banner with simulation information.
    """
    # Draw banner background
    pygame.draw.rect(screen, (20, 20, 20), (0, 0, width, dashboard_size - 5)) # -5 to leave a small gap
    pygame.draw.line(screen, (150, 150, 150), (0, dashboard_size - 5), (width, dashboard_size - 5), 2)

    # Calculate elapsed time (Real time since launch)
    elapsed_seconds = simulated_time_ms // 1000
    minutes = elapsed_seconds // 60
    seconds = elapsed_seconds % 60
    time_txt = f"{minutes:02d}:{seconds:02d}" # Format 00:00

    # Information to display
    fps = int(clock.get_fps())
    food_mode = params.get("FOOD_MODE", 1)
    if food_mode == 1:
        mode_txt = "Photosynthesis" 
    elif food_mode == 2:
        mode_txt = "Feeding" 
    else:
        mode_txt = "Mixed"
    
    # Render texts
    txt_test = font.render(f"TEST: {params['TEST_NAME']}", True, (255, 255, 255))
    txt_pop  = font.render(f"POPULATION: {pop}", True, (0, 255, 100) if pop > 0 else (255, 50, 50))
    txt_step = font.render(f"STEPS: {total_steps}", True, (200, 200, 200))
    txt_mode = font.render(f"MODE: {mode_txt}", True, (100, 200, 255))
    txt_neurons = font.render(f"AVG NODES: {mean_nodes:.1f}", True, (200, 150, 255))
    txt_fps  = font.render(f"FPS: {fps}", True, (255, 255, 0))
    txt_time = font.render(f"TIME: {time_txt}", True, (255, 255, 255))

    # Positioning on the banner
    screen.blit(txt_test, (20, 15))
    screen.blit(txt_mode, (250*1.4, 15))
    screen.blit(txt_step, (250*2, 15))
    screen.blit(txt_pop,  (250*2.5, 15))
    screen.blit(txt_neurons, (250*3.25, 15))
    screen.blit(txt_time, (250*4, 15))
    screen.blit(txt_fps,  (width-100, 15))

def show_graphics_off(screen, font, width: int, height: int, is_paused: bool, dashboard_size: int):
    """
    Displays the screen when rendering is disabled to save compute resources.
    """
    screen.fill((15, 15, 15)) # Very dark background
    txt_off = font.render(" RENDERING DISABLED (Intensive Compute) - Press 'G' to reactivate ", True, (150, 150, 150))
    screen.blit(txt_off, (width // 2 - txt_off.get_width() // 2, height // 2)) # Center the text
    if is_paused: 
        graphics_pause(screen, font, dashboard_size)
    pygame.display.flip() 

def graphics_pause(screen, font, dashboard_size: int):
    """
    Displays the pause text.
    """
    # Yellow text with a dark gray background to stand out without blinding
    txt_pause = font.render(" SIMULATION PAUSED (Press Space to resume) ", True, (255, 200, 0), (40, 40, 40))
    screen.blit(txt_pause, (20, dashboard_size + 15))