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
                console.log(data);
                var prob = data.should_bse * 100;
                $("#bse_prob").text(prob.toFixed(2)+"%");
                $("#image_origin").attr("src",data.origin)
                $("#image_bse").attr("src",data.bse)
                $("#image_heatmap").attr("src",data.heatmap)
                var html = '<tr><th>Bệnh</th><th>Xác suất mắc bệnh</th></tr>'
                Object.keys(data.results).forEach(function(e) {
                    var path = e;
                    var predict = data.results[e] * 100;
                    html += '<tr><td>'+path+'</td><td>'+predict.toFixed(2)+'%</td></tr>'
                })
                $("table#result").html(html);
                $('#result').show();
            },
        });
    });
});
