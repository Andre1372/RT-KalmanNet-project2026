function y = generate_constrained_line(C, k, T) % GENERATE_CONSTRAINED_LIN…  interval
    %
    % INPUTS:
    % C  - Starting value (quota), the value of y at t=1
    % k  - Slope (pendenza); y(t) = C + k*(t-1)
    % T  - Number of time steps
    %
    % OUTPUTS:
    % y  - Line values vector, length T (column vector)

    % 1. Time vector creation
    t = (1:T)';

    % 2. Signal generation: straight line starting at C with slope k
    y = C + k * (t - 1);

    % 3. Safety check: the whole line must stay within [0, 1]
    if any(y < 0) || any(y > 1)
        error(['Error: with C=%.4g and k=%.4g over T=%d steps, y ranges from ' ...
               '%.4g to %.4g, which falls outside [0, 1].'], C, k, T, min(y), max(y));
    end

    % 4. If the function is called without requesting outputs, show an automatic plot
    if nargout == 0
        figure('Name', 'Constrained Line Visualization');
        plot(t, y, 'LineWidth', 1.5, 'Color', '#0072BD');
        hold on;

        yline(0, 'r--', 'Lower Limit (0)', 'LabelHorizontalAlignment', 'left');
        yline(1, 'r--', 'Upper Limit (1)', 'LabelHorizontalAlignment', 'left');
        yline(C, 'g:', 'Start (C)', 'LabelHorizontalAlignment', 'left');

        ylim([-0.1, 1.1]);
        xlabel('Time step');
        ylabel('Amplitude');
        title(sprintf('Line: C = %.2f, k = %.4g', C, k));
        grid on;
        hold off;
    end
end