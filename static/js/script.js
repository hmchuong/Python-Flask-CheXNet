$(function() {
    $(document).ajaxStart(function(){
        $("#loading").css("display", "block");
    });

    $(document).ajaxComplete(function(){
        $("#loading").css("display", "none");
    });
    $('#result').hide();
    $('#upload-file-btn').click(function() {
        var form_data = new FormData($('#upload-file')[0]);
        $.ajax({
            type: 'POST',
            url: '/',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function(data) {
                var prob = data.should_bse * 100;
                $("#bse_prob").text(prob.toFixed(2)+"%");
                $('#result').show();
            },
        });
    });
});
