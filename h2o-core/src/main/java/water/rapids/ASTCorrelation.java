package water.rapids;

import water.Key;
import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;

import java.util.Arrays;

/** Correlation between columns of a frame
 TODO Handle NA's properly...
 * */
class ASTCorrelation extends ASTPrim {
    @Override
    public String[] args() { return new String[]{"ary", "x","y", "symmetric"}; }
    @Override int nargs() { return 1+3; /* (cor X Y symmetric) */}
    @Override public String str() { return "cor"; }
    @Override
    public Val apply(Env env, Env.StackHelp stk, AST asts[]) {
        Frame frx = stk.track(asts[1].exec(env)).getFrame();
        Frame fry = stk.track(asts[2].exec(env)).getFrame();
        if( frx.numRows() != fry.numRows() )
            throw new IllegalArgumentException("Frames must have the same number of rows, found "+frx.numRows()+" and "+fry.numRows());
        boolean symmetric = asts[3].exec(env).getNum()==1;
        return fry.numRows() == 1 ? scalar(frx,fry) : array(frx,fry,symmetric);
    }

    // Correlation for 1 row. Will return a scalar value.
    private ValNum scalar( Frame frx, Frame fry) {
        if( frx.numCols() != fry.numCols())
            throw new IllegalArgumentException("Single rows must have the same number of columns, found "+frx.numCols()+" and "+fry.numCols());
        Vec vecxs[] = frx.vecs();
        Vec vecys[] = fry.vecs();
        double xmean=0, ymean=0, xvar=0, yvar=0,xsd=0,ysd=0, ncols = frx.numCols(), NACount=0, xval, yval, ss=0;
        for( int r = 0; r < ncols; r++) {
            xval = vecxs[r].at(0);
            yval = vecys[r].at(0);
            if (Double.isNaN(xval) || Double.isNaN(yval))
                NACount++;
            else {
                xmean += xval;
                ymean += yval;
            }
        }
        xmean /= (ncols - NACount); ymean /= (ncols - NACount);

        for( int r = 0; r < ncols; r++ ) {
            xval = vecxs[r].at(0);
            yval = vecys[r].at(0);
            if (!(Double.isNaN(xval) || Double.isNaN(yval)))
                //Compute variance of x and y vars
                xvar += Math.pow((vecxs[r].at(0) - xmean), 2);
                yvar += Math.pow((vecys[r].at(0) - ymean), 2);
                //Compute sum of squares of x and y
                ss += (vecxs[r].at(0) - xmean) * (vecys[r].at(0) - ymean);
        }
        xsd = Math.sqrt(xvar/(frx.numRows())); //Sample Standard Deviation
        ysd = Math.sqrt(yvar/(fry.numRows())); //Sample Standard Deviation
        double cor_denom = xsd * ysd;
        return new ValNum(ss/cor_denom); //Pearson's Correlation Coefficient
    }

    // Matrix correlation  Compute correlation between all columns from each Frame
    // against each other.  Return a matrix of correlations which is frx.numCols
    // wide and fry.numCols tall.
    private Val array( Frame frx, Frame fry, boolean symmetric) {
        Vec[] vecxs = frx.vecs();
        int ncolx = vecxs.length;
        Vec[] vecys = fry.vecs();
        int ncoly = vecys.length;

        CoVarTaskEverything[] cvs = new CoVarTaskEverything[ncoly];

        double[] xmeans = new double[ncolx];
        double[] xsigma = new double[ncolx];
        for (int x = 0; x < ncoly; x++) {
            xmeans[x] = vecxs[x].mean();
            xsigma[x] = vecxs[x].sigma();
        }

        if (symmetric) {
            //1-col returns scalar
            if (ncoly == 1)
                return new ValNum(vecys[0].naCnt() == 0 ? vecys[0].sigma() : Double.NaN);

            int[] idx = new int[ncoly];
            for (int y = 1; y < ncoly; y++) idx[y] = y;
            int[] first_index = new int[]{0};
            //compute correlations between column_i and column_i+1, column_i+2, ...
            Frame reduced_fr;
            for (int y = 0; y < ncoly-1; y++) {
                idx = ArrayUtils.removeIds(idx, first_index);
                reduced_fr = new Frame(frx.vecs(idx));
                cvs[y] = new CoVarTaskEverything(vecys[y].mean(),xmeans, xsigma,vecys[y].sigma()).dfork(new Frame(vecys[y]).add(reduced_fr));
            }

            double[][] res_array = new double[ncoly][ncoly];

            //fill in the diagonals
            for (int y = 0; y < ncoly; y++)
                res_array[y][y] = vecys[y].naCnt() == 0 ? vecys[y].sigma() * vecys[y].sigma() : Double.NaN;;


//            //arrange the results into the bottom left of res_array. each successive cvs is 1 smaller in length
            for (int y = 0; y < ncoly - 1; y++)
                System.arraycopy(ArrayUtils.div(cvs[y].getResult()._cors, (fry.numRows() - 1)), 0, res_array[y], y + 1, ncoly - y - 1);

            //copy over the bottom left of res_array to its top right
            for (int y = 0; y < ncoly - 1; y++) {
                for (int x = y + 1; x < ncoly; x++) {
                    res_array[x][y] = res_array[y][x];
                }
            }
            //set Frame
            Vec[] res = new Vec[ncoly];
            Key<Vec>[] keys = Vec.VectorGroup.VG_LEN1.addVecs(ncoly);
            for (int y = 0; y < ncoly; y++) {
                res[y] = Vec.makeVec(res_array[y], keys[y]);
            }
            return new ValFrame(new Frame(fry._names, res));
        }

        // Launch tasks; each does all Xs vs one Y
        for (int y = 0; y < ncoly; y++)
            cvs[y] = new CoVarTaskEverything(vecys[y].mean(),xmeans, xsigma,vecys[y].sigma()).dfork(new Frame(vecys[y]).add(frx));

        // 1-col returns scalar
        if (ncolx == 1 && ncoly == 1) {
            return new ValNum(cvs[0].getResult()._cors[0] / (fry.numRows() - 1));
        }

        // Gather all the Xs-vs-Y correlation arrays; divide by rows
        Vec[] res = new Vec[ncoly];
        Key<Vec>[] keys = Vec.VectorGroup.VG_LEN1.addVecs(ncoly);
        for (int y = 0; y < ncoly; y++)
            res[y] = Vec.makeVec(ArrayUtils.div(cvs[y].getResult()._cors, (fry.numRows() - 1)), keys[y]);

        return new ValFrame(new Frame(fry._names, res));
    }

    private static class CoVarTaskEverything extends MRTask<CoVarTaskEverything> {
        double[] _cors;
        final double _xmeans[], _xsigma[], _ymean, _ysigma;
        CoVarTaskEverything(double ymean, double[] xmeans, double[] xsigma, double ysigma) { _ymean = ymean;
                            _xmeans = xmeans;_xsigma = xsigma; _ysigma = ysigma; }
        @Override public void map( Chunk cs[] ) {
            final int ncolsx = cs.length-1;
            final Chunk cy = cs[0];
            final int len = cy._len;
            _cors = new double[ncolsx];
            double sum;
            for( int x=0; x<ncolsx; x++ ) {
                sum = 0;
                final Chunk cx = cs[x+1];
                final double xmean = _xmeans[x];
                final double xsigma = _xsigma[x];
                for( int row=0; row<len; row++ ){
                    sum += (cx.atd(row)-xmean)*(cy.atd(row)-_ymean);
                }
                _cors[x] = sum/(xsigma * _ysigma);
            }
        }
        @Override public void reduce( CoVarTaskEverything cvt ) { ArrayUtils.add(_cors,cvt._cors); }
    }
}
