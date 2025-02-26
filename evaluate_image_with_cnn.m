function brisque_score = evaluate_image_with_cnn(image_path)
    % 計算 BRISQUE 分數
    brisque_score = compute_brisque(image_path);
    
    % 返回 BRISQUE 分數
    fprintf('BRISQUE Score for CNN input: %.4f\n', brisque_score);
end
