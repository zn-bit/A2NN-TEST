//=========================
// Engineer:  Ning Zhang
// Description:  加法核运算(向量内积，把乘法换成加法【ker参数生成时以及取负（取补码）了】并求负绝对值)

module inner_product_adder #(
    parameter IC               = 32  ,//输入并行度
    parameter PIX_W            = 8   ,//输入特征图位宽
    parameter KER_W            = 4   ,//加法卷积核位宽

    parameter PE_DATA_W        = PIX_W + 1 + $clog2(IC) //pe计算结果的位宽
)(
    input                               clk                 ,   
    // input                               ce                  ,
    input                               rst                 ,  
    input   [IC-1:0][PIX_W-1:0]         pix_vector          ,
    input                               pix_vld             ,
    input   [IC-1:0][KER_W-1:0]         ker_vector          ,
    input                               ker_vld             ,
    output  reg [PE_DATA_W-1:0]         pe_result            
    // output  reg                         pe_result_vld       
);  

localparam ADD_W = (PIX_W>=KER_W)?  PIX_W +1 : KER_W +1;
localparam DELAY_ADDERTREE = $clog2(IC);
localparam DELAY_ADD = 2;
localparam DELAY = DELAY_ADDERTREE + DELAY_ADD;

reg signed [IC-1:0][ADD_W-1:0] add_pix_ker;
reg signed [IC-1:0][ADD_W-1:0] add_pix_ker_nabs;

genvar i;
generate
    for(i=0;i<IC;i=i+1) begin:gen_add_pix_ker
        always @(posedge clk) begin
            if(pix_vld && ker_vld)
                add_pix_ker[i] <= $signed(pix_vector[i]) + $signed(ker_vector[i]);
            else
                add_pix_ker[i] <= 'd0;
        end
        always @(posedge clk ) begin
            if($signed(add_pix_ker[i])>0)
                add_pix_ker_nabs[i] <= -add_pix_ker[i];
            else
                add_pix_ker_nabs[i] <= add_pix_ker[i];
        end
    end
endgenerate

adder_tree #(                   //latency = log2(INPUTS_NUM) = 5
    .INPUTS_NUM  (IC        ),
    .IDATA_WIDTH (ADD_W     ) 
)u_adder_tree(
    .clk   (clk              ),
    .nrst  (~rst             ),
    .idata (add_pix_ker_nabs ),
    .odata (pe_result        )
);

// delay#(
//       .WIDTH   (1               ), //bit width of data, min = 1
//       .DEPTH   (DELAY           ), //depth of delay, min = 0
//       .INIT    (0               )  //init value of output
//     )delay_bias0
//     (
//       .clk     (clk                ), //input clock
//       .ce      (1'b1               ), //input clock enable
//       .din     (pix_vld && ker_vld ), //input data
//       .dout    (pe_result_vld      )  //output data
//     );


endmodule