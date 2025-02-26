function qualityscore = compute_brisque(image_path)
    % 讀取影像
    image = imread(image_path);
    
    % 確保影像為灰階
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    
    % 轉換影像為 double
    image = double(image);
    
    % 提取 BRISQUE 特徵
    features = compute_brisque_features(image);
    
    % -----------------------------------------------------------
    % **使用 SVM 進行品質評估**
    % -----------------------------------------------------------
    
    % 1. 建立 test_ind 檔案
    fid = fopen('test_ind', 'w');
    fprintf(fid, '1 ');
    for kk = 1:length(features)
        fprintf(fid, '%d:%f ', kk, features(kk));
    end
    fprintf(fid, '\n');
    fclose(fid);

    % 2. 執行 `svm-scale`
    system('svm-scale -r allrange test_ind > test_ind_scaled');

    % 3. 執行 `svm-predict`
    system('svm-predict -b 1 test_ind_scaled allmodel output > dump');

    % 4. 讀取 output 檔案
    load output
    qualityscore = output;  % **確保回傳分數，而非只顯示**
end

%----------------------------------------------------
% **內建函數：計算 BRISQUE 特徵**
%----------------------------------------------------
function feat = compute_brisque_features(imdist)
    % 設定參數
    scalenum = 2;
    window = fspecial('gaussian', 7, 7/6);
    window = window / sum(window(:));

    feat = [];

    for itr_scale = 1:scalenum
        % 計算影像亮度均值與標準差
        mu = filter2(window, imdist, 'same');
        mu_sq = mu .* mu;
        sigma = sqrt(abs(filter2(window, imdist .* imdist, 'same') - mu_sq));
        structdis = (imdist - mu) ./ (sigma + 1);

        % 提取 GGD 參數
        [alpha, overallstd] = estimateggdparam(structdis(:));
        feat = [feat, alpha, overallstd^2];

        % 提取 AGGD 參數
        shifts = [0 1; 1 0; 1 1; -1 1];

        for itr_shift = 1:4
            shifted_structdis = circshift(structdis, shifts(itr_shift, :));
            pair = structdis(:) .* shifted_structdis(:);
            [alpha, leftstd, rightstd] = estimateaggdparam(pair);
            const = sqrt(gamma(1/alpha)) / sqrt(gamma(3/alpha));
            meanparam = (rightstd - leftstd) * (gamma(2/alpha) / gamma(1/alpha)) * const;
            feat = [feat, alpha, meanparam, leftstd^2, rightstd^2];
        end

        % 影像降尺度
        imdist = imresize(imdist, 0.5);
    end
end

%----------------------------------------------------
% **內建函數：估計 AGGD 參數**
%----------------------------------------------------
function [alpha, leftstd, rightstd] = estimateaggdparam(vec)
    gam = 0.2:0.001:10;
    r_gam = ((gamma(2 ./ gam)) .^ 2) ./ (gamma(1 ./ gam) .* gamma(3 ./ gam));

    leftstd = sqrt(mean((vec(vec < 0)) .^ 2));
    rightstd = sqrt(mean((vec(vec > 0)) .^ 2));
    gammahat = leftstd / rightstd;
    rhat = (mean(abs(vec)))^2 / mean((vec).^2);
    rhatnorm = (rhat * (gammahat^3 + 1) * (gammahat + 1)) / ((gammahat^2 + 1)^2);

    [~, array_position] = min((r_gam - rhatnorm) .^ 2);
    alpha = gam(array_position);
end

%----------------------------------------------------
% **內建函數：估計 GGD 參數**
%----------------------------------------------------
function [gamparam, sigma] = estimateggdparam(vec)
    gam = 0.2:0.001:10;
    r_gam = (gamma(1 ./ gam) .* gamma(3 ./ gam)) ./ ((gamma(2 ./ gam)) .^ 2);

    sigma_sq = mean((vec) .^ 2);
    sigma = sqrt(sigma_sq);
    E = mean(abs(vec));
    rho = sigma_sq / E^2;

    [~, array_position] = min(abs(rho - r_gam));
    gamparam = gam(array_position);
end
