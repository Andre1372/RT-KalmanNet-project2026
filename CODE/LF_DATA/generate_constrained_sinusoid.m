function y = generate_constrained_sinusoid(C, k, omega, phi, T)    % GENERATE_CONSTRAINED_SINUSOID Creates a sinusoid constrained of T steps               % new
    %
    % INPUTS:
    % C     - Center (vertical offset), in the interval [0, 1]
    % k     - Amplitude scale factor, in the interval [0, 1]
    % omega - Angular frequency (rad/step)
    % phi   - Initial phase (rad)
    % T     - Number of Time steps
   
    % OUTPUTS:
    % y     - Sinusoid values vector

    % 1. Safety check on constraints for C and k
    if C < 0 || C > 1
        error('Error: Parameter C must be between 0 and 1.');
    end
    if k < 0 || k > 1
        error('Error: Parameter k must be between 0 and 1.');
    end

    % 2. Amplitude calculation to respect the limits
    A_max = min(C, 1 - C); % Maximum theoretical amplitude for this center
    A = k * A_max;         % Actual scaled amplitude

    % 3. Time vector creation
    t = (1 : T)';              % trasposto così è una colonna e c zorzi l'ha inizializzato come colonna             % new

    % 4. Signal generation
    y = C + A * sin(omega * t + phi);

    % 5. If the function is called without requesting outputs, show an automatic plot
    if nargout == 0
        figure('Name', 'Constrained Sinusoid Visualization');
        plot(t, y, 'LineWidth', 1.5, 'Color', '#0072BD');
        hold on;
        
        % Adding reference lines
        yline(0, 'r--', 'Lower Limit (0)', 'LabelHorizontalAlignment', 'left');
        yline(1, 'r--', 'Upper Limit (1)', 'LabelHorizontalAlignment', 'left');
        yline(C, 'g:', 'Center (C)', 'LabelHorizontalAlignment', 'left');
        
        % Plot formatting
        ylim([-0.1, 1.1]);
        xlabel('Time Steps');                                                                       % new
        ylabel('Amplitude');
        title(sprintf('Sinusoid: C = %.2f, k = %.2f  ->  Calculated Amplitude (A) = %.3f', C, k, A));
        grid on;
        hold off;
    end
end
